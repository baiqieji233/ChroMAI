"""Optional Beat This! raw beat/downbeat logits (50 Hz), never downloads implicitly."""
from __future__ import annotations
import hashlib
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import tempfile
import threading
import weakref
from functools import lru_cache
from types import MethodType
import numpy as np

_hashes = {}
_models = {}
_model_lock = threading.Lock()
_frame_locks=weakref.WeakValueDictionary()
_frame_registry_lock=threading.Lock()
_identity_lock=threading.RLock()
_identities={}
_identity_generation=0


@lru_cache(maxsize=1)
def _package_version():
    return importlib.metadata.version('beat-this')


def raw_cache_key(audio_digest,identity):
    import json
    return hashlib.sha256(json.dumps({'schema':'composer_beat_this_raw_v4',
        'audio_sha256':str(audio_digest),'backend_identity':identity},sort_keys=True,
        separators=(',',':')).encode()).hexdigest()


def validate_backend_identity(config):
    """Inference must use the same analyzer identity that supplied training."""
    if config.structure_policy not in {'metrical_context_v1','metrical_context_v2'}:return
    current=backend_identity(config.structure_backend,refresh=True)
    expected=getattr(config,'structure_backend_identity',None)
    if expected is None or expected!=current:
        raise ValueError('Structure analysis backend identity differs from training; restore its exact weights/version or explicitly migrate into a new training directory')


def _stable_attend(self,q,k,v):
    """Bounded explicit attention, local to the optional inference model.

    Avoid changing global SDP flags while several GPU jobs are running.
    Chunking queries preserves attention over ALL keys, without introducing
    segment boundaries or a different attention mask.
    """
    import torch
    if self.training:raise RuntimeError('Structure attention adapter is inference-only')
    scale=self.scale if self.scale is not None else q.shape[-1]**-.5
    keys=k.float().transpose(-2,-1);values=v.float()
    chunks=[]
    for start in range(0,q.shape[-2],128):
        scores=(q[...,start:start+128,:].float()*scale)@keys
        chunks.append((scores.softmax(dim=-1)@values).to(v.dtype))
    return torch.cat(chunks,dim=-2)

def checkpoint_path():
    explicit = os.environ.get('MAICHART_BEAT_THIS_CHECKPOINT')
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get('TORCH_HOME', Path.home()/'.cache'/'torch'))/'hub'/'checkpoints'/'beat_this-final0.ckpt'

def backend_cache_token():
    """Cheap process-local invalidation token; no filesystem queries."""
    return (_identity_generation,os.environ.get('MAICHART_BEAT_THIS_CHECKPOINT'),os.environ.get('TORCH_HOME'))


def clear_backend_identity_cache():
    global _identity_generation
    with _identity_lock:
        _identities.clear()
        _identity_generation+=1


def backend_identity(requested,*,refresh=False):
    """Freeze analyzer identity for a run; refresh only at run boundaries.

    Cache hits perform no package discovery, weight stat, path resolution or
    JSON hashing. Return a copy so callers cannot mutate the pinned identity.
    """
    global _identity_generation
    key=(requested,os.environ.get('MAICHART_BEAT_THIS_CHECKPOINT'),os.environ.get('TORCH_HOME'))
    with _identity_lock:
        old=_identities.get(key)
        if old is not None and not refresh:return dict(old)
        if refresh:_package_version.cache_clear()
        current=_discover_backend_identity(requested,force_hash=refresh)
        if old is not None and current!=old:_identity_generation+=1
        _identities[key]=current
        return dict(current)


def _discover_backend_identity(requested,*,force_hash=False):
    if requested not in ('auto', 'native', 'beat_this'):
        raise ValueError('Supported structure backends: auto, native, beat_this')
    if requested == 'native': return {'name':'native_multicycle_v1'}
    p = checkpoint_path()
    reason = None
    if importlib.util.find_spec('beat_this') is None:
        reason = 'beat_this package is not installed'
    elif not p.is_file():
        reason = f'local checkpoint does not exist: {p}'
    else:
        try:
            from beat_this.inference import Audio2Frames  # Verify imports, without loading weights.
        except (ImportError, OSError, RuntimeError) as exc:
            reason = f'beat_this dependencies are not usable: {type(exc).__name__}: {exc}'
    if reason:
        if requested == 'beat_this':
            raise RuntimeError(f'BeatThis is unavailable: {reason}. Install dependencies and provide a local checkpoint via MAICHART_BEAT_THIS_CHECKPOINT.')
        return {'name':'native_multicycle_v1', 'fallback':reason}
    stat = p.stat(); key = (str(p.resolve()), stat.st_size, stat.st_mtime_ns)
    if force_hash or key not in _hashes:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
        _hashes[key]=h.hexdigest()
    return {'name':'beat_this_raw_logits_v1', 'checkpoint_sha256':_hashes[key], 'fps':50,
            'waveform_policy':'native_mono_float32_v1','package_version':_package_version(),
            'attention_policy':'query_chunked_explicit_math_v1'}

def beat_this_frames(audio,cache_dir,identity,*,device='cpu'):
    if identity['name']!='beat_this_raw_logits_v1':return None
    from maichart.form.rational_audio import _source_digest
    digest=(getattr(audio,'metadata',{}) or {}).get('audio_sha256') or _source_digest(Path(audio.source),cache_dir)
    key=(str(Path(cache_dir).resolve()) if cache_dir else '',raw_cache_key(digest,identity))
    with _frame_registry_lock:
        lock=_frame_locks.get(key)
        if lock is None:lock=threading.RLock();_frame_locks[key]=lock
    with lock:return _beat_this_frames(audio,cache_dir,identity,device=device,source_digest=digest)


def _beat_this_frames(audio, cache_dir, identity, *, device='cpu',source_digest=None):
    """Return read-only [frames,2] raw logits; checkpoint path is local only."""
    if identity['name'] != 'beat_this_raw_logits_v1': return None
    device = os.environ.get('MAICHART_BEAT_THIS_DEVICE', device)
    source = Path(audio.source)
    if not source.is_file(): raise FileNotFoundError(f'BeatThis audio source unavailable: {source}')
    digest = source_digest or (getattr(audio, 'metadata', {}) or {}).get('audio_sha256')
    if not digest:
        h = hashlib.sha256()
        with source.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
        digest = h.hexdigest()
    key=raw_cache_key(digest,identity)
    path=Path(cache_dir)/'composer_beat_this_v1'/(key+'.npy') if cache_dir is not None else None
    if path and path.exists():
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.ndim==2 and a.shape[1]==2: return a
    import torch
    from beat_this.inference import Audio2Frames
    from maichart.form.waveform_cache import mapped_waveform
    # Use the exact same decoder/cache as native acoustic features. A second
    # MP3 decoding path must not introduce an independent timing offset.
    signal,sr=mapped_waveform(source,str(digest),cache_dir)
    if not len(signal):raise ValueError('Empty audio source')
    model_key=(str(checkpoint_path().resolve()), str(device), identity['checkpoint_sha256'],threading.get_ident())
    with _model_lock:
        if model_key not in _models:
            instance=Audio2Frames(checkpoint_path=str(checkpoint_path().resolve()), device=device)
            from beat_this.model.roformer import Attend
            for module in instance.model.modules():
                if isinstance(module,Attend):module.forward=MethodType(_stable_attend,module)
            _models[model_key]=instance
        model = _models[model_key]
    with torch.inference_mode():
        if str(device).startswith('cuda'):
            stream=torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream):beat,downbeat=model(signal,sr)
            stream.synchronize()
        else:beat,downbeat=model(signal,sr)
    a=np.column_stack((beat.detach().cpu().numpy(), downbeat.detach().cpu().numpy())).astype(np.float32)
    if not np.isfinite(a).all(): raise ValueError('Non-finite BeatThis logits')
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix=key+'.',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f: np.save(f,a,allow_pickle=False); f.flush(); os.fsync(f.fileno())
            try:os.replace(tmp,path)
            except PermissionError:
                # A concurrent process may already have published and mapped
                # this immutable key. Never delete an open Windows mapping.
                existing=np.load(path,mmap_mode='r',allow_pickle=False)
                if existing.shape!=a.shape or existing.dtype!=a.dtype or not np.isfinite(existing[[0,-1]]).all():raise
                return existing
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    a.setflags(write=False)
    return a


def resolve_structure_backend(requested='auto'):
    """Resolve once at run creation, then persist the returned value in config.

    Persisting this value prevents newly installed weights from changing an
    existing run on resume. Resolution never downloads weights.
    """
    info = backend_identity(requested,refresh=True)
    return 'beat_this' if info['name'] == 'beat_this_raw_logits_v1' else 'native'
