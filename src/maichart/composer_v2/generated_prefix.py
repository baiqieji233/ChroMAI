"""Load reviewed, offline sampled histories for a paired Composer v2 fine-tuning experiment."""
import argparse
from bisect import bisect_left
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from maichart.event_timing import EventPoint, EventTempo, parse_event_chart
from maichart.generation.sustain_state import SustainStateIndex


SCHEMA = 'chromai-generated-prefix-v1'
MODES = ('off', 'authored', 'generated')
_SHA256 = re.compile('[0-9a-f]{64}')
_TOLERANCE = Fraction(1, 6)


@dataclass(frozen=True)
class Sample:
    chart_sha256: str
    audio_sha256: str
    slot: int
    start: Fraction
    join: Fraction
    end: Fraction
    prefix: tuple[EventPoint, ...]

    @property
    def chart_key(self):
        return self.chart_sha256, self.slot


@dataclass(frozen=True)
class Bank:
    samples: tuple[Sample, ...]
    sha256: str

    @property
    def charts(self):
        return {sample.chart_key for sample in self.samples}

    def contract(self, mode):
        return dict(mode=mode, sha256=self.sha256)


@dataclass(frozen=True)
class View:
    song: object
    known: list[EventPoint]


def read_bank(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or not payload.get('samples'):
        raise ValueError('生成前文样本文件没有有效的 v1 样本')
    samples = []
    for number, row in enumerate(payload['samples'], 1):
        try:
            chart = row['chart_sha256']
            audio = row['audio_sha256']
            slot = row['difficulty']
            start, join, end = (Fraction(row[key]) for key in ('start_beat', 'join_beat', 'end_beat'))
            prefix = tuple(EventPoint(Fraction(p['beat']), p['note_text']) for p in row['prefix'])
            if not (_SHA256.fullmatch(chart) and _SHA256.fullmatch(audio) and type(slot) is int and 4 <= slot <= 6
                    and 0 <= start < join <= end and len(prefix) >= 4
                    and all(isinstance(p.note_text, str) and p.note_text.strip() and start <= p.beat < join for p in prefix)
                    and all(a.beat < b.beat for a, b in zip(prefix, prefix[1:]))):
                raise ValueError('无效的身份、拍位、顺序或前文')
            samples.append(Sample(chart, audio, slot, start, join, end, prefix))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f'生成前文样本第 {number} 条无效：{exc}') from exc
    keys = {(s.chart_sha256, s.audio_sha256, s.slot, s.start, s.join, s.end) for s in samples}
    if len(keys) != len(samples):
        raise ValueError('生成前文样本包含重复接缝')
    return Bank(tuple(samples), hashlib.sha256(raw).hexdigest())


def validate_mode(mode, sequence_policy, initialized, resume):
    if mode not in MODES:
        raise ValueError('未知的生成前文实验模式')
    if mode != 'off' and (sequence_policy != 'continuous_events_v1' or not (initialized or resume)):
        raise ValueError('生成前文实验需要使用已有连续事件模型加强训练（finetune）')


def _active(state, beat, second):
    return sorted((m['type'], str(m['touch_pad'] or m['lane']), str(m['end_lane']),
                   m['slide_track_key'], float(m['end_second']))
                  for m in state._candidate_markers(second)
                  if m['source_beat'] < beat and float(m['end_second']) > second + .02
                  and not m['is_mine'] and not m['track_mine'])


def _matched_sustain(authored, sampled, cut, second):
    real, replay = _active(authored, cut, second), _active(sampled, cut, second)
    return len(real) == len(replay) and all(
        a[:-1] == b[:-1] and abs(a[-1] - b[-1]) <= .1 for a, b in zip(real, replay))


def _views(dataset, song, sample):
    from .generate import _check_window, _new_guard

    pts, beats = song.points, song.point_beats
    start, begin, last = (bisect_left(beats, beat) for beat in (sample.start, sample.join, sample.end))
    if (begin >= len(pts) or last >= len(pts) or beats[begin] != sample.join or beats[last] != sample.end
            or begin - start < 4 or last - begin < 3):
        raise ValueError('接缝两侧需要至少四个真人事件，目标的首尾必须是原谱事件')
    lead = pts[start:begin]
    if abs(sample.prefix[-1].beat - lead[-1].beat) > _TOLERANCE or list(sample.prefix) == lead:
        raise ValueError('生成前文收尾与真人拍位错开超过 1/6 拍，或前文完全相同')
    suffix = pts[begin:last+1]
    base = pts[:start]
    altered = [*base, *sample.prefix]
    tempo = [EventTempo(beat, bpm) for beat, bpm in zip(song.clock.beats, song.clock.bpms)]
    first = song.clock.seconds[0]
    real_state = song.sustain if song.sustain is not None else SustainStateIndex(pts, tempo, first)
    sampled_state = SustainStateIndex(altered, tempo, first)
    cut = (max(lead[-1].beat, sample.prefix[-1].beat) + sample.join) / 2
    if not _matched_sustain(real_state, sampled_state, cut, song.clock.at(cut)):
        raise ValueError('接缝仍在持续的动作与原谱不一致')
    guard = _new_guard(tempo, first, song.audio.duration_sec)
    for point in base:
        guard.commit(point)
    for prefix in (lead, sample.prefix):
        trial, reason = _check_window(guard, [*prefix, *suffix])
        if trial is None:
            raise ValueError(f'真人后续未通过连续可行性检查（{reason}）')
    return begin, last+1, View(song, pts[:begin]), View(replace(song, sustain=sampled_state), altered)


def item(dataset, view, song_id, begin, end):
    original = dataset.songs[song_id]
    out = dataset.build_stream(view.song, view.known, view.known[-1].beat,
                               target=original.points[begin:end], training=True, reference_start=None)
    out['curve_teacher_allowed'] = False
    dataset._writing_targets(out, dataset._configuration(song_id), original, begin, end)
    return out


def install(dataset, bank, mode):
    """Use the same two views and weights per phrase in both arms."""
    mapping = {}
    for index, song in enumerate(dataset.songs):
        key = song.chart_sha256, song.difficulty, song.audio.identity.get('audio_sha256')
        mapping.setdefault(key, index)
    windows, ranges, lengths, views = [], [], [], []
    for number, sample in enumerate(bank.samples, 1):
        try:
            si = mapping.get((*sample.chart_key, sample.audio_sha256))
            if si is None:
                raise ValueError('训练清单没有通过当前筛选且同时匹配音频和谱面的记录')
            begin, end, authored, sampled = _views(dataset, dataset.songs[si], sample)
            left = item(dataset, authored, si, begin, end)
            right = item(dataset, sampled, si, begin, end)
            a, b = left['labels'] != -100, right['labels'] != -100
            if (not left['labels'][a].equal(right['labels'][b])
                    or not left['token_weights'][a].equal(right['token_weights'][b])
                    or not left['sample_weight'].equal(right['sample_weight'])
                    or not (left['token_weights'] * left['sample_weight']).any()):
                raise ValueError('两组目标或标注权重不一致，或没有可训练的目标')
            windows.extend([(si, begin)] * 2)
            ranges.extend([end] * 2)
            lengths.extend([len(left['input_ids'])] * 2)
            views.extend([authored, sampled if mode == 'generated' else authored])
        except (OSError, ValueError, KeyError, IndexError) as exc:
            raise ValueError(f'生成前文样本第 {number} 条无法用于训练：{exc}') from exc
    dataset.windows, dataset.ranges, dataset.lengths = windows, ranges, lengths
    dataset.generated_prefix_views = views
    dataset.audit.update(windows=len(windows), generated_prefix_experiment={
        **bank.contract(mode), 'phrases': len(bank.samples), 'views': len(windows)})
    dataset.audit['timing_recovery'] = {'policy': 'off', 'views': 0}


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='加入一段已经审阅的模型前文。起止单位为谱面拍，可写分数。')
    for field in ('reference', 'generated', 'audio', 'output'):
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--difficulty', type=int, choices=(4, 5, 6), default=5)
    parser.add_argument('--generated-difficulty', type=int, choices=(4, 5, 6), default=5)
    for field in ('start-beat', 'join-beat', 'end-beat'):
        parser.add_argument('--' + field, required=True, type=Fraction)
    args = parser.parse_args(argv)
    original = parse_event_chart(args.reference, args.difficulty, strict_notes=True)
    generated = parse_event_chart(args.generated, args.generated_difficulty, strict_notes=True)
    if original.tempo_changes != generated.tempo_changes or abs(original.audio_first - generated.audio_first) > 1e-5:
        parser.error('真人谱和生成谱需要使用相同拍网和首拍；请先确认音频对应')
    sample = dict(chart_sha256=_file_hash(args.reference), audio_sha256=_file_hash(args.audio),
                  difficulty=args.difficulty, start_beat=str(args.start_beat),
                  join_beat=str(args.join_beat), end_beat=str(args.end_beat),
                  prefix=[dict(beat=str(p.beat), note_text=p.note_text) for p in generated.points
                          if args.start_beat <= p.beat < args.join_beat])
    saved = read_bank(args.output) if args.output.exists() else None
    existing = json.loads(args.output.read_text(encoding='utf-8-sig'))['samples'] if saved else []
    if not (0 <= args.start_beat < args.join_beat <= args.end_beat and len(sample['prefix']) >= 4):
        parser.error('请填写有效拍位范围，生成前文需要至少四个事件')
    if any(all(row[key] == sample[key] for key in ('chart_sha256', 'audio_sha256', 'difficulty',
                                                  'start_beat', 'join_beat', 'end_beat')) for row in existing):
        parser.error('该接缝已经在样本文件中')
    temp = args.output.with_name(args.output.name + '.tmp')
    temp.write_text(json.dumps(dict(schema=SCHEMA, samples=[*existing, sample]), ensure_ascii=False, indent=2),
                    encoding='utf-8')
    temp.replace(args.output)


if __name__ == '__main__':
    main()
