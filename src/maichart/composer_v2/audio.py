"""Offline, content-addressed audio conditioning; training reads only numpy maps.

MERT is optional at import time. ``native`` is explicitly mel-only and always
marks semantic features unavailable; a failed MERT extraction never falls back.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from weakref import ReferenceType, ref
from zipfile import BadZipFile

import numpy as np

from maichart.composer.runtime_memory import cache_capacity, host_memory
from maichart.form.cache_support import source_digests

MODEL_ID = 'm-a-p/MERT-v1-330M'
MODEL_REVISION = '5240c2708a5acaee1007f43fb9735c7dcd0b78c9'
SAMPLE_RATE = 24000
MODEL_CACHE = Path(__file__).resolve().parents[3] / 'cache' / 'composer_v2' / 'models'
SCHEMA = 'composer_v2_audio_v1'
MAX_OPEN_BANKS = 512
MAX_MAPPED_BYTES = 256 * 1024 * 1024
_VALIDATION_CACHE_SIZE = 4096
_BANK_ARRAYS = ('mel', 'mel_times_sec', 'semantic', 'semantic_times_sec')
_BANK_FILES = ('index.json', *(key + '.npy' for key in _BANK_ARRAYS))


class _SemanticLayerReady(Exception):
    """Unwind MERT once the requested layer has produced its output."""


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _interpolate(values, times, seconds):
    seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if not len(times):
        return np.zeros((len(seconds), values.shape[1]), dtype=np.float32)
    right = np.searchsorted(times, seconds, side='right')
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    weight = np.clip((seconds - times[left]) / np.maximum(times[right] - times[left], 1e-12), 0., 1.)
    blended = np.asarray(values[left], dtype=np.float32) * (1. - weight[:, None])
    blended += np.asarray(values[right], dtype=np.float32) * weight[:, None]
    return blended.astype(np.float32)


@dataclass(frozen=True)
class AudioFeatures:
    mel: np.ndarray
    mel_times_sec: np.ndarray
    semantic: np.ndarray
    semantic_times_sec: np.ndarray
    duration_sec: float
    identity: dict
    _source_signature: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def semantic_available(self):
        return self.identity['backend'] == 'mert'

    def sample(self, seconds):
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
        if not np.isfinite(seconds).all():
            raise ValueError('Audio query times must be finite')
        mel = _interpolate(self.mel, self.mel_times_sec, seconds)
        semantic = _interpolate(self.semantic, self.semantic_times_sec, seconds)
        valid = ((seconds >= 0.) & (seconds <= self.duration_sec)
                 & bool(self.semantic_available))
        return mel, semantic, valid

    def sample_for_training(self, seconds):
        """Read compact adjacent semantic frames for device-side interpolation."""
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
        if not np.isfinite(seconds).all():
            raise ValueError('Audio query times must be finite')
        times = self.semantic_times_sec
        if not len(times):
            mel, semantic, valid = self.sample(seconds)
            return mel, semantic, semantic.copy(), np.zeros(len(seconds), dtype=np.float64), valid
        right = np.searchsorted(times, seconds, side='right')
        left = np.clip(right - 1, 0, len(times) - 1)
        right = np.clip(right, 0, len(times) - 1)
        weight = np.clip((seconds - times[left]) / np.maximum(times[right] - times[left], 1e-12), 0., 1.)
        valid = ((seconds >= 0.) & (seconds <= self.duration_sec) & bool(self.semantic_available))
        return (_interpolate(self.mel, self.mel_times_sec, seconds),
                self.semantic[left], self.semantic[right], weight, valid)

    def summary(self, max_tokens=160):
        """Time-binned musical tokens, preserving the song's actual timeline."""
        if int(max_tokens) <= 0:
            raise ValueError('max_tokens must be positive')
        count = min(int(max_tokens), max(1, int(np.ceil(self.duration_sec))))
        edges = np.linspace(0., self.duration_sec, count + 1)
        times = (edges[:-1] + edges[1:]) * .5
        mel, semantic, valid = self.sample(times)
        # Pool each real-time interval rather than just sampling its midpoint.
        for values, source_times, target in ((self.mel, self.mel_times_sec, mel),
                                              (self.semantic, self.semantic_times_sec, semantic)):
            boundaries = np.searchsorted(source_times, edges)
            boundaries[-1] = np.searchsorted(source_times, edges[-1], side='right')
            for index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                if hi > lo:
                    target[index] = np.asarray(values[lo:hi], dtype=np.float32).mean(axis=0)
        # Eight spectral-band standard deviations and four onset time cells
        # (mean and peak) retain within-bin changes erased by mean pooling.
        detail = np.zeros((count, 16), dtype=np.float32)
        boundaries = np.searchsorted(self.mel_times_sec, edges)
        boundaries[-1] = np.searchsorted(self.mel_times_sec, edges[-1], side='right')
        for index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if hi <= lo:
                continue
            values = np.asarray(self.mel[lo:hi], dtype=np.float32)
            bands = values[:, :96].reshape(len(values), 8, 12).mean(-1)
            detail[index, :8] = np.log1p(bands.std(0))
            # Flux uses the preceding bank frame too, preserving boundary attacks.
            previous = np.asarray(self.mel[max(0, lo-1):hi-1], dtype=np.float32)
            if len(previous) != len(values):
                previous = np.concatenate((values[:1], values[:-1]), axis=0)
            flux = np.maximum(values[:, :96]-previous[:, :96], 0.).mean(-1)
            cells = np.minimum(3, ((self.mel_times_sec[lo:hi]-edges[index]) /
                                   max(edges[index+1]-edges[index], 1e-12)*4).astype(int))
            for cell in range(4):
                selected = flux[cells == cell]
                if len(selected):
                    detail[index, 8+cell] = np.log1p(selected.mean())
                    detail[index, 12+cell] = np.log1p(selected.max())
        return dict(mel=mel, semantic=semantic, semantic_valid=valid, detail=detail,
                    times_sec=times.astype(np.float32),
                    position=(times / max(self.duration_sec, 1e-12)).astype(np.float32))


@dataclass(frozen=True)
class AudioBankReference:
    """Keep the bank's identity without pinning every song's file mappings."""

    store: AudioFeatureStore
    directory: Path
    duration_sec: float
    identity: dict
    _prepared: ReferenceType[AudioFeatures] | None = field(default=None, repr=False, compare=False)

    def _features(self):
        return self.store._mapped(self.directory, self.identity)

    def sample(self, seconds):
        return self._features().sample(seconds)

    def sample_for_training(self, seconds):
        return self._features().sample_for_training(seconds)

    def summary(self, max_tokens=160):
        prepared = self._prepared() if self._prepared is not None else None
        return self.store._summary(self.directory, self.identity, self.duration_sec, max_tokens, prepared)

    @property
    def semantic_available(self):
        return self.identity['backend'] == 'mert'

    @property
    def mel(self):
        return self._features().mel

    @property
    def mel_times_sec(self):
        return self._features().mel_times_sec

    @property
    def semantic(self):
        return self._features().semantic

    @property
    def semantic_times_sec(self):
        return self._features().semantic_times_sec


class AudioFeatureStore:
    def __init__(self, cache_dir, backend='mert', model_path=None, device='cpu', *, max_mapped_bytes=None):
        if backend not in ('mert', 'native'):
            raise ValueError('Audio backend must be mert or native')
        self.cache_dir = Path(cache_dir)
        from .acceleration import resolve_audio_device
        self.backend = backend
        self.device = resolve_audio_device(str(device)) if backend == 'mert' else 'cpu'
        self.model_path = Path(model_path).resolve() if model_path else None
        self._model = self._processor = None
        self._openvino = None
        self._acceleration_failed = False
        self._model_identity = None
        self._banks = OrderedDict()
        self._validated_banks = OrderedDict()
        self._digest_root = str(self.cache_dir.resolve())
        self._source_digests = None
        self.max_mapped_bytes = MAX_MAPPED_BYTES if max_mapped_bytes is None else int(max_mapped_bytes)
        if self.max_mapped_bytes < 0:
            raise ValueError('Audio mapping budget must be nonnegative')
        self._effective_mapped_bytes = self.max_mapped_bytes
        self._memory_check_at = 0.
        self._mapped_bytes = 0
        self._spectrum = self._mel_scale = None
        self.cache_hits = self.cache_builds = self.bank_opens = 0
        self.map_hits = self.evictions = self.bank_validations = self.summary_hits = 0

    def _identity(self, audio_path, *, remember_source=False):
        if self._model_identity is None:
            if self.backend == 'native':
                self._model_identity = None
            elif self.model_path is None:
                self._model_identity = dict(repo=MODEL_ID, revision=MODEL_REVISION)
            else:
                if not self.model_path.is_dir():
                    raise FileNotFoundError(f'MERT model directory missing: {self.model_path}')
                files = sorted(p for p in self.model_path.rglob('*') if p.is_file()
                               and p.suffix in ('.json', '.py', '.bin', '.safetensors'))
                if not files:
                    raise ValueError('MERT model directory contains no model files')
                self._model_identity = dict(local_files={str(p.relative_to(self.model_path)): _digest(p) for p in files})
        if self._source_digests is None and (remember_source or
                (Path(self._digest_root) / 'source_digests.sqlite3').is_file()):
            self._source_digests = source_digests(self._digest_root)
        audio_hash = (self._source_digests.digest(audio_path) if self._source_digests is not None
                      else _digest(audio_path))
        return dict(schema=SCHEMA, backend=self.backend, audio_sha256=audio_hash,
                    model=self._model_identity, sample_rate=SAMPLE_RATE,
                    mono='native_channel_mean_float32', resample='torchaudio_sinc_interp_hann_default',
                    mel=dict(n_fft=1024, hop=240, bands=96, center=True, power=2., log='log1p'),
                    semantic=dict(layer=8, hop=320, receptive_field=400, center_samples=199.5,
                                  chunk_frames=225, context_frames=75, dtype='float16',
                                  weightnorm_compat='legacy_gv_to_parametrizations_v1'))

    def _directory(self, identity):
        return self.cache_dir / 'audio_v1' / _json_hash(identity)

    def load(self, audio_path):
        identity = self._identity(Path(audio_path).resolve())
        return self._mapped(self._directory(identity), identity)

    def reference(self, features):
        return AudioBankReference(self, self._directory(features.identity),
                                  features.duration_sec, features.identity, ref(features))

    def _mapped(self, directory, identity):
        now = time.monotonic()
        if now >= self._memory_check_at:
            self._memory_check_at = now + 1.
            self._effective_mapped_bytes = cache_capacity(self.max_mapped_bytes, self._mapped_bytes,
                                                        host_memory())
            self._trim_mappings()
        cached = self._banks.get(directory)
        if cached is not None:
            self.map_hits += 1
            self._banks.move_to_end(directory)
            return cached[0]
        features = self._read(directory, identity)
        self.bank_opens += 1
        size = sum(getattr(features, key).nbytes for key in _BANK_ARRAYS)
        if size <= self._effective_mapped_bytes:
            self._banks[directory] = (features, size)
            self._mapped_bytes += size
            self._trim_mappings()
        return features

    def _trim_mappings(self):
        while len(self._banks) > MAX_OPEN_BANKS or self._mapped_bytes > self._effective_mapped_bytes:
            _, (_, evicted_size) = self._banks.popitem(last=False)
            self._mapped_bytes -= evicted_size
            self.evictions += 1

    def mapping_stats(self):
        return dict(bank_opens=self.bank_opens, open_banks=len(self._banks),
                    mapped_bytes=self._mapped_bytes, budget_bytes=self.max_mapped_bytes,
                    effective_budget_bytes=self._effective_mapped_bytes,
                    map_hits=self.map_hits, evictions=self.evictions, bank_validations=self.bank_validations,
                    summary_hits=self.summary_hits)

    @staticmethod
    def _file_signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _bank_signature(self, directory):
        return tuple(self._file_signature(directory / name) for name in _BANK_FILES)

    def _read(self, directory, identity):
        index = directory / 'index.json'
        try:
            signature = self._bank_signature(directory)
        except FileNotFoundError as error:
            raise FileNotFoundError(f'Audio cache missing; run offline prepare first: {directory}') from error
        cached = self._validated_banks.get(directory)
        needs_validation = cached is None or cached[0] != signature
        if needs_validation:
            metadata = json.loads(index.read_text(encoding='utf-8'))
        else:
            metadata = cached[1]
        if metadata['identity'] != identity:
            raise ValueError(f'Audio cache identity mismatch: {directory}')
        arrays = {key: np.load(directory / (key + '.npy'), mmap_mode='r', allow_pickle=False)
                  for key in _BANK_ARRAYS}
        for name, dim in (('mel', 98), ('semantic', 1024)):
            values, times = arrays[name], arrays[name + '_times_sec']
            if values.ndim != 2 or values.shape[1] != dim or times.shape != (len(values),):
                raise ValueError(f'Invalid {name} cache shape: {directory}')
            if needs_validation and (not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
                raise ValueError(f'Invalid {name} cache timeline: {directory}')
        if needs_validation:
            self.bank_validations += 1
            self._validated_banks[directory] = (signature, metadata)
        self._validated_banks.move_to_end(directory)
        if len(self._validated_banks) > _VALIDATION_CACHE_SIZE:
            self._validated_banks.popitem(last=False)
        return AudioFeatures(**arrays, duration_sec=float(metadata['duration_sec']), identity=identity,
                             _source_signature=signature)

    def _summary(self, directory, identity, duration_sec, max_tokens, prepared=None):
        if int(max_tokens) != 160:
            return self._mapped(directory, identity).summary(max_tokens)
        # Derived sidecar for the fixed training summary. Existing immutable
        # feature banks acquire it on demand; feature identities stay unchanged.
        path = directory / 'summary_160_v2.npz'
        count = min(160, max(1, int(np.ceil(duration_sec))))
        fields = dict(mel=((count, 98), np.float32), semantic=((count, 1024), np.float32),
                      semantic_valid=((count,), np.bool_), times_sec=((count,), np.float32),
                      position=((count,), np.float32), detail=((count, 16), np.float32))
        bank_signature = None
        try:
            bank_signature = self._bank_signature(directory)
            signature = _json_hash(bank_signature)
            with np.load(path, allow_pickle=False) as cached:
                if cached['source_signature'].item() == signature:
                    result = {key: cached[key] for key in fields}
                    if all(array.shape == fields[key][0] and array.dtype == fields[key][1] and
                           np.isfinite(array).all() for key, array in result.items()):
                        self.summary_hits += 1
                        return result
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            pass
        if prepared is None or prepared._source_signature != bank_signature:
            entry = self._banks.get(directory)
            if entry is not None and entry[0]._source_signature != bank_signature:
                del self._banks[directory]
                self._mapped_bytes -= entry[1]
                self.evictions += 1
            prepared = self._mapped(directory, identity)
        result = prepared.summary(max_tokens)
        temporary = None
        try:
            if bank_signature is None or self._bank_signature(directory) != bank_signature:
                return result
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, source_signature=np.asarray(signature), **result)
            temporary.replace(path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return result

    def _encoder(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            kwargs = dict(trust_remote_code=True)
            source = str(self.model_path) if self.model_path else MODEL_ID
            if self.model_path is None:
                kwargs.update(revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE))
            processor = Wav2Vec2FeatureExtractor.from_pretrained(source, **kwargs)
            model = self._load_first_eight(source, kwargs)
            if model is not None:
                if processor.sampling_rate != SAMPLE_RATE or model.config.hidden_size != 1024:
                    raise ValueError('Expected MERT 330M with 24kHz and 1024 hidden dimensions')
                target = 'cpu' if self.device.startswith('openvino-') else self.device
                self._model = model.requires_grad_(False).eval().to(target)
                self._processor = processor
                return self._model, self._processor
            model, loading = AutoModel.from_pretrained(source, output_loading_info=True, **kwargs)
            # Torch >=2 exposes weight_norm through parametrizations. The
            # original MERT checkpoint uses the legacy weight_g/weight_v keys;
            # Transformers 4.39 otherwise silently leaves positional conv random.
            missing = set(loading.get('missing_keys', []))
            unexpected = set(loading.get('unexpected_keys', []))
            prefix = 'encoder.pos_conv_embed.conv.'
            modern = {prefix + 'parametrizations.weight.original0',
                      prefix + 'parametrizations.weight.original1'}
            legacy = {prefix + 'weight_g', prefix + 'weight_v'}
            if missing == modern and unexpected == legacy and not loading.get('mismatched_keys'):
                if self.model_path is None:
                    from huggingface_hub import hf_hub_download
                    weight_path = hf_hub_download(MODEL_ID, 'pytorch_model.bin',
                                                 revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE))
                else:
                    weight_path = self.model_path / 'pytorch_model.bin'
                state = torch.load(weight_path, map_location='cpu', weights_only=True)
                state[prefix + 'parametrizations.weight.original0'] = state.pop(prefix + 'weight_g')
                state[prefix + 'parametrizations.weight.original1'] = state.pop(prefix + 'weight_v')
                model.load_state_dict(state, strict=True)
                del state
                logging.getLogger(__name__).warning(
                    'MERT compatibility repair complete: mapped legacy positional-convolution '
                    'weight_g/weight_v to parametrizations original0/original1; full checkpoint '
                    'strict=True reload succeeded with no missing or unexpected weights. '
                    'The preceding Transformers initialization warnings have been resolved.')
            elif missing or unexpected or loading.get('mismatched_keys') or loading.get('error_msgs'):
                raise ValueError(f'MERT checkpoint did not load exactly: {loading}')
            if processor.sampling_rate != SAMPLE_RATE or model.config.hidden_size != 1024:
                raise ValueError('Expected MERT 330M with 24kHz and 1024 hidden dimensions')
            target = 'cpu' if self.device.startswith('openvino-') else self.device
            model.requires_grad_(False).eval().to(target)
            # Only block 8 is ever read. Discard unused blocks after strict load.
            model.encoder.layers = torch.nn.ModuleList(list(model.encoder.layers[:8]))
            self._model, self._processor = model, processor
        return self._model, self._processor

    def _load_first_eight(self, source, kwargs):
        """Map just the needed blocks, avoiding initialization/copy of 24 blocks."""
        import torch
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(source, **kwargs)
        # DeepNorm's scaling depends on the original depth. Retain the generic
        # loader for such custom models and for safetensors-only local sources.
        if getattr(config, 'deepnorm', False):
            return None
        if self.model_path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(MODEL_ID, 'pytorch_model.bin', revision=MODEL_REVISION,
                                   cache_dir=str(MODEL_CACHE))
        else:
            path = self.model_path / 'pytorch_model.bin'
            if not path.is_file():
                return None
        config.num_hidden_layers = 8
        mapping = getattr(config, 'auto_map', {})
        model_class = None
        if 'AutoModel' in mapping:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            code_kwargs = {k:v for k,v in kwargs.items() if k in ('revision','cache_dir')}
            model_class = get_class_from_dynamic_module(mapping['AutoModel'], source, **code_kwargs)
        # No random initialization or allocation of weights that will be replaced.
        with torch.device('meta'):
            model = model_class(config) if model_class is not None else AutoModel.from_config(config, trust_remote_code=True)
        try:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except RuntimeError as exc:
            if 'mmap' not in str(exc).lower():
                raise
            state = torch.load(path, map_location='cpu', weights_only=True)
        selected = {}
        prefix = 'encoder.pos_conv_embed.conv.'
        aliases = {prefix+'weight_g':prefix+'parametrizations.weight.original0',
                   prefix+'weight_v':prefix+'parametrizations.weight.original1'}
        for name, value in state.items():
            if name.startswith('encoder.layers.') and int(name.split('.')[2]) >= 8:
                continue
            key = aliases.get(name, name)
            if key in selected:
                raise ValueError('Duplicate MERT positional-convolution weights: '+key)
            selected[key] = value
        model.load_state_dict(selected, strict=True, assign=True)
        return model

    def _semantic_processor(self):
        if self._processor is None:
            from transformers import Wav2Vec2FeatureExtractor
            kwargs = dict(trust_remote_code=True)
            source = str(self.model_path) if self.model_path else MODEL_ID
            if self.model_path is None:
                kwargs.update(revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE))
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(source, **kwargs)
            if self._processor.sampling_rate != SAMPLE_RATE:
                raise ValueError('Expected MERT with 24kHz input')
        return self._processor

    def prepare(self, audio_path):
        source = Path(audio_path).resolve()
        identity = self._identity(source, remember_source=True)
        directory = self._directory(identity)
        if (directory / 'index.json').is_file():
            features = self._mapped(directory, identity)
            self.cache_hits += 1
            return features
        import torch
        import torchaudio
        from maichart.form.waveform_cache import mapped_waveform

        # The existing native-rate waveform cache is shared across processes.
        wave, rate = mapped_waveform(source, sha256=identity['audio_sha256'], cache_dir=self.cache_dir)
        signal = torch.from_numpy(np.array(wave, dtype=np.float32, copy=True))
        if not len(signal):
            raise ValueError(f'Empty audio: {source}')
        duration = len(signal) / rate
        if rate != SAMPLE_RATE:
            signal = torchaudio.functional.resample(signal, rate, SAMPLE_RATE)
        if len(signal) < 400:
            signal = torch.nn.functional.pad(signal, (0, 400 - len(signal)))
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.audio-', dir=directory.parent))
        try:
            # Spectrogram extraction is CPU-only; no encoder remains in Dataset.
            if self._spectrum is None:
                self._spectrum = torchaudio.transforms.Spectrogram(
                    n_fft=1024, hop_length=240, power=2., pad_mode='constant')
                self._mel_scale = torchaudio.transforms.MelScale(
                    n_mels=96, sample_rate=SAMPLE_RATE, n_stft=513)
            spectrum = self._spectrum(signal)
            bands = self._mel_scale(spectrum)
            logmel = torch.log1p(bands).T
            flux = torch.zeros(len(logmel))
            flux[1:] = (logmel[1:] - logmel[:-1]).clamp_min(0).mean(dim=-1)
            energy = torch.log1p(spectrum.mean(dim=0))
            mel = torch.cat((logmel, flux[:, None], energy[:, None]), dim=-1).numpy()
            mel_times = np.arange(len(mel), dtype=np.float64) * .01
            keep = mel_times <= duration
            np.save(temporary / 'mel.npy', mel[keep].astype(np.float32), allow_pickle=False)
            np.save(temporary / 'mel_times_sec.npy', mel_times[keep], allow_pickle=False)
            if self.backend == 'mert':
                self._extract_semantic(signal, duration, temporary)
            else:
                np.save(temporary / 'semantic.npy', np.zeros((1, 1024), dtype=np.float16), allow_pickle=False)
                np.save(temporary / 'semantic_times_sec.npy', np.array([0.], dtype=np.float64), allow_pickle=False)
            metadata = dict(identity=identity, duration_sec=duration,
                            runtime=dict(torch=torch.__version__, torchaudio=torchaudio.__version__,
                                         semantic_device=self.device,
                                         acceleration_fallback=self._acceleration_failed))
            (temporary / 'index.json').write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')
            # Publish only complete immutable banks. Concurrent winner is safe.
            try:
                temporary.rename(directory)
            except OSError:
                if not (directory / 'index.json').is_file():
                    raise
            features = self._mapped(directory, identity)
            self.cache_builds += 1
            return features
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _extract_semantic(self, signal, duration, directory):
        import torch
        processor = self._semantic_processor()
        model = hook = None
        if self.device.startswith('openvino-') and not self._acceleration_failed:
            try:
                if self._openvino is None:
                    from .openvino_audio import OpenVINOMERT
                    self._openvino = OpenVINOMERT(self.cache_dir, self._model_identity, self.device, self._encoder)
            except Exception:
                logging.getLogger(__name__).exception('MERT OpenVINO initialization failed; using CPU')
                self._acceleration_failed = True
        count = max(1, (len(signal) - 400) // 320 + 1)
        times = (np.arange(count, dtype=np.float64) * 320 + 199.5) / SAMPLE_RATE
        keep_count = max(1, int(np.searchsorted(times, duration, side='right')))
        bank = np.lib.format.open_memmap(directory / 'semantic.npy', mode='w+', dtype='float16',
                                        shape=(keep_count, 1024))
        # Capture the same block-8 tensor and stop before evaluating blocks 9–24.
        captured = []
        def capture(_module, _inputs, outputs):
            captured.append(outputs[0].detach())
            raise _SemanticLayerReady
        try:
            with torch.inference_mode():
                for first in range(0, keep_count, 225):
                    last = min(first + 225, keep_count)
                    left, right = max(0, first - 75), min(count, last + 75)
                    chunk = signal[left * 320:(right - 1) * 320 + 400].numpy()
                    inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors='pt')
                    captured.clear()
                    encoded = None
                    if self._openvino is not None and not self._acceleration_failed:
                        try:
                            encoded = self._openvino.infer(inputs['input_values'].numpy())
                        except Exception:
                            logging.getLogger(__name__).exception('MERT OpenVINO inference failed; using CPU')
                            self._acceleration_failed = True
                    if encoded is None:
                        if model is None:
                            model, _ = self._encoder()
                        hook = model.encoder.layers[7].register_forward_hook(capture)
                        target = next(model.parameters()).device
                        inputs = {key: value.to(target) for key, value in inputs.items()}
                        try:
                            model(**inputs, output_hidden_states=False)
                        except _SemanticLayerReady:
                            pass
                        finally:
                            hook.remove()
                            hook = None
                        if len(captured) != 1:
                            raise RuntimeError('MERT block 8 capture failed')
                        selected = captured[0][0, first-left:last-left]
                        values = selected.float().cpu().numpy()
                    else:
                        values = encoded[0, first-left:last-left]
                    if values.shape != (last-first, 1024):
                        raise RuntimeError('MERT output timeline differs from the pinned convolution contract')
                    if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
                        raise ValueError('MERT produced invalid float16 cache values')
                    bank[first:last] = values
                    captured.clear()
                    del inputs
            bank.flush()
        finally:
            if hook is not None:
                hook.remove()
            captured.clear()
            del bank
        np.save(directory / 'semantic_times_sec.npy', times[:keep_count], allow_pickle=False)

    def close(self):
        """Release preparation resources before the training loader starts."""
        self._banks.clear()
        self._mapped_bytes = 0
        self._memory_check_at = 0.
        had_encoder = self._model is not None
        self._model = self._processor = None
        self._openvino = None
        self._spectrum = self._mel_scale = None
        if not had_encoder:
            return
        import gc
        gc.collect()
        if self.device.startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
