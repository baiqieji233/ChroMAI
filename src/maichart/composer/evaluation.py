"""Reproducible, audio-disjoint Composer evaluation helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


STRUCTURE_COMPARISON_KEYS=('rhythm_echo_rate','layout_echo_rate','ternary_rate','ternary_isolated_rate',
                           'aa_rate','pingpong_rate','double_chain_rate')


def compare_chart_reports(reference,generated,*,factor=2.):
    """Flag descriptive deviations; a reference is one valid composition."""
    pairs=[]
    for key in ('touch_heads','touch_events','events_3plus_heads'):
        left=reference.get('rates',{}).get(key,{})
        right=generated.get('rates',{}).get(key,{})
        pairs.append((key,left.get('rate'),right.get('rate'),
                      left.get('denominator'),right.get('denominator')))
    for key in ('same','ABA','ABAB'):
        def value(report):
            return report.get('configuration',{}).get('repetition',{}).get('exact_event_text',{}).get('rates',{}).get(key)
        pairs.append(('exact_event_'+key,value(reference),value(generated),None,None))
    alerts=[]
    for key,base,current,base_n,current_n in pairs:
        if base is None or current is None:continue
        if current>base*factor and current>0:
            alerts.append({'metric':key,'reference':base,'generated':current,
                'multiple':current/base if base>0 else None,
                'reference_denominator':base_n,'generated_denominator':current_n,
                'reason':'over_reference_factor' if base>0 else 'nonzero_against_zero_reference'})
    result={'factor':factor,'alerts':alerts,
        'scope':'review signals only; reference choice and small denominators can explain deviations'}
    if reference.get('structure') and generated.get('structure'):
        # Same-song structure: bar echo, rhythm grids, repetition. Descriptive
        # only; one song can lack triplets or echoes without being wrong.
        result['structure']={key:{'reference':reference['structure'].get(key),'generated':generated['structure'].get(key)}
            for key in STRUCTURE_COMPARISON_KEYS}
    return result


def form_chart_eligibility(difficulty):
    """Metadata gate matching FormDataset's default slot and numeric-level range.

    None means eligible; a stable reason names a metadata exclusion. This does
    not claim the chart can be parsed or survives preparation-time filtering.
    """
    from maichart.training.event_dataset import numeric_level
    if not isinstance(difficulty, dict): return 'invalid_difficulty'
    try: slot = int(difficulty.get('difficulty_index', difficulty.get('index', 0)))
    except (TypeError, ValueError, OverflowError): return 'invalid_slot'
    if not 4 <= slot <= 6: return 'slot_out_of_range'
    try: level = numeric_level(difficulty.get('level_raw', difficulty.get('level')))
    except (TypeError, ValueError, OverflowError): return 'invalid_level'
    if not 12. <= level <= 15.: return 'level_out_of_range'
    return None


def eligible_manifest_songs(songs, chart_eligibility):
    """Filter copies only, retaining source indices and exclusion evidence."""
    from collections import Counter
    eligible=[]; indices=[]; reasons=Counter(); excluded=Counter(); charts=0
    for index, song in enumerate(songs):
        if not isinstance(song, dict):
            excluded['invalid_song'] += 1; continue
        difficulties=song.get('difficulties', [])
        if not isinstance(difficulties, list):
            excluded['invalid_difficulties'] += 1; continue
        selected=[]; song_reasons=set()
        for difficulty in difficulties:
            reason=chart_eligibility(difficulty)
            if reason is None: selected.append(difficulty)
            else: reasons[reason] += 1; song_reasons.add(reason)
        if not selected:
            excluded['no_eligible_difficulty'] += 1
            for reason in song_reasons or {'no_difficulties'}: excluded[reason] += 1
            continue
        eligible.append({**song, 'difficulties': selected}); indices.append(index); charts += len(selected)
    return eligible, dict(source_song_count=len(songs),eligible_song_count=len(eligible),
        excluded_song_count=len(songs)-len(eligible),eligible_chart_count=charts,
        excluded_difficulty_reason_counts=dict(reasons),excluded_song_reason_counts=dict(excluded),
        eligible_source_indices=indices,
        scope='metadata only; chart parsing and preparation may reject further charts')


def split_manifest(manifest, output_dir, *, fraction=.05, seed=0, cache_dir=None, chart_eligibility=None,
                   audio_digest=None):
    """Create train/dev/final manifests, joining identical audio and song IDs.

    ``fraction`` is the total fraction of groups assigned to both holdouts.
    Each holdout gets at least one group. Small datasets can therefore exceed
    the requested fraction; the report records actual group and song ratios. All audio
    files must exist so that an unverifiable path cannot masquerade as unseen.
    """
    from maichart.training.dataset_catalog import _source_asset, _source_root
    source = Path(manifest).resolve()
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    songs = data.get('songs')
    if not isinstance(songs, list) or not songs:
        raise ValueError('manifest must contain a nonempty songs list')
    eligibility_report = None
    if chart_eligibility is not None:
        songs, eligibility_report = eligible_manifest_songs(songs, chart_eligibility)
        if not songs: raise ValueError('No songs with eligible chart metadata for automatic holdout')
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between 0 and 1')
    digest_audio = audio_digest or file_hash
    if audio_digest is None and cache_dir is not None:
        from maichart.form.cache_support import source_digests
        digest_audio = source_digests(str(Path(cache_dir).resolve())).digest
    parents = list(range(len(songs)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    identities = []
    for i, song in enumerate(songs):
        if not isinstance(song, dict) or not song.get('audio_path'):
            raise ValueError(f'song {i} lacks audio_path')
        audio = _source_asset(song['audio_path'], source, data)
        if not audio.is_file():
            raise FileNotFoundError(f'cannot verify audio identity: {audio}')
        digest = digest_audio(audio)
        keys = [('audio_sha256', digest)]
        for field in ('song_id', 'music_id', 'maidata_path'):
            if song.get(field) is not None:
                value = str(song[field])
                if field == 'maidata_path':
                    value = str(_source_asset(value, source, data)).casefold()
                keys.append((field, value))
        music_key = _normalized_music_key(song)
        if music_key is not None:
            keys.append(music_key)
        for key in keys:
            if key in seen:
                a, b = find(i), find(seen[key])
                parents[a] = b
            else:
                seen[key] = i
        identities.append({'audio_path': str(audio), 'audio_sha256': digest})
    groups = {}
    for i in range(len(songs)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 3:
        raise ValueError('at least three independent song/audio groups are required')
    ordered = sorted(groups.values(), key=lambda g: sorted(identities[i]['audio_sha256'] for i in g))
    random.Random(seed).shuffle(ordered)
    full_difficulties = [[d for d in song.get('difficulties', ())
                          if d.get('learning_role', song.get('learning_role', 'full')) != 'mechanics']
                         for song in songs]
    has_roles = any(len(keep) != len(song.get('difficulties', ())) for keep, song in zip(full_difficulties, songs))
    eligible_groups = len(ordered)
    if has_roles:
        full = [g for g in ordered if any(full_difficulties[i] for i in g)]
        if len(full) < 3:
            raise ValueError('自动划分需要至少三组独立录音含「完整学习」谱面；也可以指定独立验证清单。')
        ordered = full + [g for g in ordered if not any(full_difficulties[i] for i in g)]
        eligible_groups = len(full)
    holdout_count = min(eligible_groups - 1, max(2, round(eligible_groups * fraction)))
    dev_count = holdout_count // 2
    assignments = {'dev': ordered[:dev_count], 'final': ordered[dev_count:holdout_count],
                   'train': ordered[holdout_count:]}
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'composer-song-audio-split-v1', 'source_manifest': str(source),
              'source_sha256': file_hash(source), 'requested_total_holdout_fraction': fraction, 'seed': seed,
              'group_count': len(groups), 'actual_total_holdout_group_fraction': holdout_count / len(groups),
              'splits': {}}
    if eligibility_report is not None: report['chart_eligibility'] = eligibility_report
    for name, selected in assignments.items():
        indices = sorted(i for g in selected for i in g)
        if has_roles and name != 'train':
            indices = [i for i in indices if full_difficulties[i]]
        output = dict(data)
        # Relative paths resolve exactly as they did at the original location.
        output['source_root'] = str(_source_root(source, data))
        output['songs'] = []
        for i in indices:
            entry = dict(songs[i])
            if has_roles and name != 'train':
                entry['difficulties'] = full_difficulties[i]
            for key in ('audio_path', 'maidata_path'):
                if entry.get(key):
                    entry[key] = str(_source_asset(entry[key], source, data))
            output['songs'].append(entry)
        output['song_count']=len(output['songs'])
        difficulties=[d for song in output['songs'] for d in song.get('difficulties',[])]
        output['difficulty_count']=len(difficulties)
        output['usable_difficulty_count']=sum(bool(d.get('event_usable_for_training')) for d in difficulties)
        if 'summary' in output:output['source_manifest_summary']=output.pop('summary')
        output['split_metadata']={'role':name,'source_manifest':str(source),'source_sha256':report['source_sha256'],
            'groups':len(selected),'seed':seed,'requested_total_holdout_fraction':fraction}
        path = out / f'{name}_manifest.json'
        if path.resolve() == source:
            raise ValueError('output must not overwrite the source manifest')
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report['splits'][name] = {'path': str(path), 'sha256': file_hash(path),
            'groups': len(selected), 'songs': len(indices),
            'group_fraction': len(selected) / len(groups), 'song_fraction': len(indices) / len(songs),
            'audio_sha256': sorted({identities[i]['audio_sha256'] for i in indices})}
    report['actual_total_holdout_song_fraction'] = sum(
        report['splits'][name]['songs'] for name in ('dev','final')
    ) / len(songs)
    report_path = out / 'split_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report['report_path'] = str(report_path)
    return report


_GENERIC_MUSIC_FIELDS = frozenset({"", "-", "unknown", "none", "n/a", "na", "null", "tba", "variousartists"})


def _normalized_music_key(song: dict) -> tuple[str, str] | None:
    """Group cross-encoded MP3/OGG copies of the same (title, artist) song without colliding bare titles."""
    import re
    import unicodedata
    raw_title = song.get("title")
    raw_artist = song.get("artist")
    if not isinstance(raw_title, str) or not isinstance(raw_artist, str):
        return None
    norm_title = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", raw_title).casefold())
    norm_artist = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", raw_artist).casefold())
    if len(norm_title) < 2 or len(norm_artist) < 2:
        return None
    if norm_title in _GENERIC_MUSIC_FIELDS or norm_artist in _GENERIC_MUSIC_FIELDS:
        return None
    return ("music_key", f"{norm_title}::{norm_artist}")
def assert_unseen(training_manifest, holdout_manifest, *, cache_dir=None, audio_digest=None):
    """Reject overlap by audio content or song identity, including old manifests."""
    from maichart.training.dataset_catalog import _source_asset
    digest_audio = audio_digest or file_hash
    if audio_digest is None and cache_dir is not None:
        from maichart.form.cache_support import source_digests
        digest_audio = source_digests(str(Path(cache_dir).resolve())).digest
    def keys(path):
        path = Path(path).resolve(); data = json.loads(path.read_text(encoding='utf-8-sig'))
        result = set()
        for i, song in enumerate(data['songs']):
            if not song.get('audio_path'):
                raise ValueError(f'missing audio_path in {path}, song {i}')
            result.add(('audio', digest_audio(_source_asset(song['audio_path'], path, data))))
            for field in ('song_id', 'music_id'):
                if song.get(field) is not None:
                    result.add((field, str(song[field])))
            if song.get('maidata_path'):
                result.add(('maidata', str(_source_asset(song['maidata_path'], path, data)).casefold()))
            music_key = _normalized_music_key(song)
            if music_key is not None:
                result.add(music_key)
        return result
    overlap = keys(training_manifest) & keys(holdout_manifest)
    if overlap:
        raise ValueError(
            f'training and holdout overlap ({len(overlap)} identities); unseen claim rejected. '
            f'训练清单：{Path(training_manifest).resolve()}；验证清单：{Path(holdout_manifest).resolve()}。'
            '这表示同曲或相同音频同时进入训练和验证，身份数不等于歌曲数。'
            '新训练请将“验证数据来源”设为“自动按歌曲划分”，或使用与该验证清单配套且互不重叠的训练清单；'
            '不要将旧开发集与包含它的全量训练清单一起使用。')
    return True


def evaluate_teacher_forced(model, dataset, *, device, batch_size=32, max_batches=0,
                            precision='fp32', loss_kwargs=None):
    """Evaluate on authored histories; pool pointer/action losses by their weights.

    Auxiliary loss means are explicitly batch means, because their masks and
    nonlinear EOS balancing differ across batches. This is not free generation.
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    from .data import collate_composer
    from .train import _forward_batch, _loss
    if precision not in ('fp32', 'bf16', 'fp16'):
        raise ValueError('precision must be fp32, bf16, or fp16')
    device = torch.device(device)
    if precision != 'fp32' and device.type != 'cuda':
        raise ValueError('mixed precision evaluation requires CUDA')
    if batch_size <= 0 or max_batches < 0:
        raise ValueError('invalid batch_size or max_batches')
    evaluation_source=dataset
    selection='all_representable_events'
    if max_batches and len(dataset)>max_batches*batch_size:
        budget=max_batches*batch_size
        groups={}
        for index,sample in enumerate(dataset.samples):
            groups.setdefault(sample.song_index,[]).append(index)
        ordered=sorted(groups)
        if len(ordered)>budget:
            ordered=[ordered[i] for i in np.linspace(0,len(ordered)-1,budget,dtype=int)]
        selected=[]
        base,remainder=divmod(budget,len(ordered))
        for group_index,key in enumerate(ordered):
            rows=groups[key];count=min(len(rows),base+int(group_index<remainder))
            selected.extend(rows[i] for i in np.linspace(0,len(rows)-1,count,dtype=int))
        evaluation_source=Subset(dataset,selected)
        selection='fixed_song_balanced_uniform_event_positions'
    loader = DataLoader(evaluation_source, batch_size=batch_size, shuffle=False, collate_fn=collate_composer)
    from .echo_metrics import EchoMetrics
    echo = EchoMetrics()
    sums = {}; masses = {}; batches = samples = 0
    modes = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate(); numpy_rng = np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            for batch in loader:
                if max_batches and batches >= max_batches: break
                tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                with torch.autocast(device_type=device.type, enabled=precision != 'fp32',
                                    dtype=torch.bfloat16 if precision == 'bf16' else torch.float16):
                    output = _forward_batch(model, tensors, packed_actions=True)
                    _, metrics = _loss(output, tensors, len(dataset.codec_atoms) + 1, **(loss_kwargs or {}))
                echo.update(model, tensors, output)
                w = tensors.get('sample_weight', torch.ones_like(tensors['eos_target']))
                transition = tensors.get('transition_weight', torch.ones_like(w))
                pointer_mass = (w * transition * tensors.get('pointer_weight', torch.ones_like(w))).sum()
                action_mass = (w * transition * tensors.get('action_weight', torch.ones_like(w))).sum()
                action_weights = w * transition * tensors.get('action_weight', torch.ones_like(w))
                targets = tensors['action_target'] if 'action_target' in tensors else None
                if targets is not None:
                    active = torch.ones_like(targets, dtype=torch.bool)
                    if targets.size(1) > 1:
                        active[:, 1:] = targets[:, :-1].ne(0).long().cumprod(-1).bool()
                    nonstop_mass = (active * targets.ne(0) * action_weights[:, None]).sum()
                    stop_mass = (active * targets.eq(0) * action_weights[:, None]).sum()
                else:
                    nonstop_mass = stop_mass = w.new_zeros(())
                # Preserve Python/double report accumulation, but read every
                # scalar in one transfer instead of synchronizing per metric.
                names = tuple(metrics)
                report_values = torch.stack([pointer_mass, action_mass, nonstop_mass, stop_mass,
                                             *(metrics[key] for key in names)]).cpu().tolist()
                pointer_mass, action_mass, nonstop_mass, stop_mass = report_values[:4]
                size = int(w.numel()); samples += size; batches += 1
                for key, v in zip(names, report_values[4:]):
                    if key.endswith(('_ce_sum', '_ce_weight')):
                        sums[key] = sums.get(key, 0.) + v
                        continue
                    mass = (nonstop_mass if key in ('action_nonstop_ce', 'audio_action_nonstop_ce') else
                            stop_mass if key in ('action_stop_ce', 'audio_action_stop_ce') else
                            pointer_mass if key in ('pointer_loss', 'audio_pointer_loss') else
                            action_mass if key in ('action_loss', 'audio_action_loss', 'action_legacy_ce', 'action_joint_nll', 'action_first_slot_ce', 'audio_action_legacy_ce', 'audio_action_joint_nll', 'audio_action_first_slot_ce') else
                            size if key == 'pointer_accuracy' else 1.)
                    sums[key] = sums.get(key, 0.) + v * mass
                    masses[key] = masses.get(key, 0.) + mass
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng)
        for module, was_training in modes:
            module.training = was_training
    if not batches: raise ValueError('evaluation dataset is empty')
    result = {k: v / masses[k] if masses[k] else None for k, v in sums.items() if k in masses}
    for key, val in sums.items():
        if key.endswith('_ce_sum'):
            base = key[:-4]; mass = sums.get(base + '_weight', 0.)
            result[base] = val / mass if mass else None
            result[base + '_weight'] = mass
    return {'schema': 'composer-teacher-forced-v1', 'samples': samples, 'batches': batches,
            'metrics': result, 'aggregation': 'pointer and sequence action losses weighted by objective mass; conditional CE pooled; remaining auxiliary metrics batch means',
            'precision': precision,'selection':selection,
            'available_events':len(dataset),
            'scope': 'authored histories; not free generation', **echo.report(model)}


def run_generation_panel(checkpoint, holdout_manifest, output_dir, *, song_limit=2,
                         seeds=(0, 1), device='auto', max_events=0):
    """Run a fixed free-generation panel on first distinct holdout audio files.

    Reference metadata and failures are recorded explicitly; no unseen claim is
    inferred from a manifest name. Temperatures are fixed across checkpoints.
    """
    import math
    import os
    import subprocess
    import sys
    import uuid
    import shutil
    from maichart.composer.checkpoint import read_payload, export_inference
    from maichart.maidata import parse_maidata_file
    from maichart.training.dataset_catalog import _source_asset

    if song_limit < 1 or max_events < 0:
        raise ValueError('song_limit must be positive and max_events nonnegative')
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and unique')
    source = Path(holdout_manifest).resolve()
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(data.get('songs'), list):
        raise ValueError('holdout manifest must contain songs')
    root = Path(__file__).resolve().parents[3]
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    run_dir = output / ('run_' + uuid.uuid4().hex)
    run_dir.mkdir()
    source_checkpoint = Path(checkpoint).resolve()
    source_hash = file_hash(source_checkpoint)
    frozen_checkpoint = run_dir / 'evaluation_checkpoint.pt'
    payload = read_payload(source_checkpoint)
    if payload.get('checkpoint_kind') == 'inference':
        shutil.copyfile(source_checkpoint, frozen_checkpoint)
    else:
        export_inference(source_checkpoint, frozen_checkpoint)
    del payload
    if file_hash(source_checkpoint) != source_hash:
        frozen_checkpoint.unlink(missing_ok=True)
        raise RuntimeError('source checkpoint changed while creating evaluation snapshot')
    frozen_hash = file_hash(frozen_checkpoint)
    report = {'schema': 'composer-generation-panel-v1', 'checkpoint': str(frozen_checkpoint),
        'checkpoint_sha256': frozen_hash, 'source_checkpoint': str(source_checkpoint),
        'source_checkpoint_sha256': source_hash, 'holdout_manifest': str(source),
        'holdout_manifest_sha256': file_hash(source), 'song_limit': song_limit,
        'seeds': list(seeds), 'device': str(device), 'max_events': max_events,
        'pointer_temperature': .5, 'action_temperature': .75,
        'unseen_verified': False, 'songs': [], 'skipped': [], 'errors': [],
        'scope': 'fixed same-song reference panel; no training or parent lineage disjointness claimed'}
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    seen = set(); chosen = 0
    for index, entry in enumerate(data['songs']):
        if chosen >= song_limit: break
        if not isinstance(entry, dict):
            report['skipped'].append({'manifest_index': index, 'reason': 'invalid song entry'})
            continue
        audio_value = entry.get('audio_path')
        if not audio_value:
            report['skipped'].append({'manifest_index': index, 'reason': 'audio path missing'})
            continue
        audio = _source_asset(audio_value, source, data)
        if not audio.is_file():
            report['skipped'].append({'manifest_index': index, 'audio': str(audio), 'reason': 'audio file missing'})
            continue
        digest = file_hash(audio)
        if digest in seen: continue
        seen.add(digest); chosen += 1
        info = {'manifest_index': index, 'song_id': entry.get('song_id'), 'audio': str(audio),
                'audio_sha256': digest}
        try:
            if not entry.get('maidata_path'): raise ValueError('reference chart path missing')
            reference = _source_asset(entry['maidata_path'], source, data)
            if not reference.is_file(): raise ValueError('reference chart file missing')
            chart = parse_maidata_file(reference)
            if chart.first is None or not str(chart.first).strip(): raise ValueError('reference first metadata missing')
            first = float(chart.first)
            if not math.isfinite(first): raise ValueError('reference first is not finite')
            blocks = {block.index: block for block in chart.difficulties if block.inote}
            difficulty = None
            for candidate in entry.get('difficulties', []):
                if not isinstance(candidate, dict): continue
                if candidate.get('event_usable_for_training') is False or candidate.get('usable_for_training') is False or candidate.get('has_chart') is False:
                    continue
                try:
                    slot = int(candidate.get('difficulty_index', candidate.get('index', 0)))
                    block = blocks[slot]
                    bpm_value = candidate.get('initial_bpm') or chart.wholebpm
                    level_value = candidate.get('level')
                    if level_value is None: level_value = block.level
                    if isinstance(level_value, str) and level_value.endswith('+'):
                        level_value = float(level_value[:-1]) + .5
                    bpm, level = float(bpm_value), float(level_value)
                    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(level): continue
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                difficulty = (slot, bpm, level); break
            if difficulty is None: raise ValueError('no valid reference difficulty with BPM and level')
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            report['skipped'].append({**info, 'reason': str(exc)})
            continue
        slot, bpm, level = difficulty
        song_output = run_dir / f'song_{index:05d}'
        cmd = [sys.executable, str(root / 'tools' / 'evaluate_composer.py'),
               '--checkpoint', str(frozen_checkpoint), '--audio', str(audio),
               '--reference', str(reference), '--difficulty', str(slot), '--output-dir', str(song_output),
               '--seeds', *(str(seed) for seed in seeds), '--device', str(device),
               '--bpm', str(bpm), '--first', str(first), '--level', str(level),
               '--pointer-temperature', '0.5', '--action-temperature', '0.75', '--max-events', str(max_events)]
        info.update(reference=str(reference), difficulty=slot, bpm=bpm, first=first, level=level, command=cmd)
        try:
            result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace')
            if result.returncode:
                report['errors'].append({**info, 'returncode': result.returncode,
                                         'stderr': result.stderr[-8000:], 'stdout': result.stdout[-4000:]})
                continue
            panel_path = song_output / 'evaluation.json'
            panel = json.loads(panel_path.read_text(encoding='utf-8'))
            if not isinstance(panel, dict) or not isinstance(panel.get('runs'), list):
                raise ValueError('generator did not produce a valid evaluation report')
            report['songs'].append({**info, 'report_path': str(panel_path), 'evaluation': panel})
            for run in panel.get('runs',[]):
                structure=run.get('reference_comparison',{}).get('structure')
                if structure:
                    print('composer_structure_observation '+json.dumps({'song_id':entry.get('song_id'),
                        'seed':run.get('seed'),**structure}),flush=True)
                for alert in run.get('reference_comparison',{}).get('alerts',[]):
                    print('composer_quality_observation '+json.dumps({'song_id':entry.get('song_id'),
                        'seed':run.get('seed'),**alert}),flush=True)
        except (OSError, ValueError, TypeError) as exc:
            report['errors'].append({**info, 'reason': str(exc)})
    report['selected_distinct_audio_count'] = chosen
    report['completed_songs'] = len(report['songs'])
    report['status'] = ('failed' if report['errors'] else 'complete' if report['songs'] else 'no_evaluable_songs')
    target = output / 'panel_report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report['report_path'] = str(target)
    return report
