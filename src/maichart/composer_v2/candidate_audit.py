"""Read-only, same-pool selection diagnostics; never creates training labels."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path


def number(value):
    try:
        value = float(Fraction(str(value)))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ranks(values):
    return [sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2 for x in values]


def correlation(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    a, b = map(ranks, zip(*pairs))
    am, bm = mean(a), mean(b)
    numerator = sum((x-am)*(y-bm) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator / denominator if denominator else None


def metrics(candidate, beats):
    diagnostics = candidate.get('diagnostics') or {}
    components = candidate.get('components') or {}
    result = {key: number(diagnostics.get(source)) for key, source in (
        ('atoms', 'atoms_in_span'), ('events', 'events_in_span'), ('tokens', 'tokens_in_span'))}
    result.update(score=number(candidate.get('score')), comfort=number(components.get('comfort')),
                  timing_support=number(components.get('timing_support')))
    result['density'] = result['atoms'] / beats if result['atoms'] is not None and beats and beats > 0 else None
    return result


def shadow(candidate, beats, event_count):
    probability = candidate.get('probability') or {}
    if probability.get('schema') != 'composer-v2-probability-v1':
        return {}, 'unavailable: probability record missing'
    if probability.get('coverage') != 'sampled_and_forced_complete':
        return {}, 'unavailable: incomplete probability coverage'
    timing, configuration = probability.get('timing') or {}, probability.get('configuration') or {}
    all_tokens = probability.get('all_tokens') or {}
    timing_sum = number(timing.get('raw_logprob_sum'))
    config_sum = number(configuration.get('raw_logprob_sum'))
    grammar_sum, tokens = number(all_tokens.get('grammar_logprob_sum')), number(all_tokens.get('tokens'))
    return dict(raw_timing_sum=timing_sum,
        raw_timing_per_beat=timing_sum / beats if timing_sum is not None and beats and beats > 0 else None,
        raw_configuration_per_event=config_sum / event_count if config_sum is not None and event_count and event_count > 0 else None,
        grammar_mean_per_token=grammar_sum / tokens if grammar_sum is not None and tokens and tokens > 0 else None), None


def delta(left, right):
    return {key: left[key] - right[key] if left[key] is not None and right[key] is not None else None for key in left}


def audit_pool(pool):
    candidates = pool.get('candidates') or []
    start, stop = number(pool.get('start_beat')), number(pool.get('end_beat'))
    beats = stop-start if start is not None and stop is not None else None
    eligible = [i for i, c in enumerate(candidates) if c.get('complete') is True and
                (number(c.get('events')) or len(c.get('rows') or [])) > 0 and
                (c.get('eligible') is True if 'eligible' in c else not c.get('reason'))]
    values = [metrics(c, beats) for c in candidates]
    selected = pool.get('selected')
    valid_selected = isinstance(selected, int) and not isinstance(selected, bool) and selected in eligible
    shadows, unavailable = {}, {}
    for i in eligible:
        event_count = values[i]['events']
        shadows[i], unavailable[i] = shadow(candidates[i], beats, event_count)
    ranking = {}
    for name in ('raw_timing_sum', 'raw_timing_per_beat', 'raw_configuration_per_event', 'grammar_mean_per_token'):
        available = [i for i in eligible if shadows[i].get(name) is not None]
        complete = bool(eligible) and len(available) == len(eligible)
        order = sorted(available, key=lambda i: (-shadows[i][name], i)) if complete else []
        ranking[name] = dict(status='available' if complete else 'unavailable', order=order,
            winner=order[0] if order else None, values={str(i): shadows[i].get(name) for i in eligible})
    others = [values[i] for i in eligible if i != selected]
    other_mean = {key: mean([v[key] for v in others]) for key in values[0]} if values else {}
    correlations = {}
    for score in ('score', 'comfort', 'timing_support', *ranking):
        scores = [values[i].get(score) if score in values[i] else shadows[i].get(score) for i in eligible]
        correlations[score] = {key: correlation(scores, [values[i][key] for i in eligible]) for key in ('density', 'events', 'tokens')}
    return dict(pool_index=pool.get('pool_index'), mode=pool.get('mode', 'unknown'), start_beat=pool.get('start_beat'), end_beat=pool.get('end_beat'),
        eligible=eligible, selected=selected, selected_eligible=valid_selected, baseline=0 if 0 in eligible else None,
        candidate_metrics=values, selected_minus_baseline=delta(values[selected], values[0]) if valid_selected and 0 in eligible else None,
        selected_minus_other_mean=delta(values[selected], other_mean) if valid_selected else None,
        shadow_rankings=ranking, probability_unavailable={str(i): r for i, r in unavailable.items() if r},
        rank_correlations=correlations)


def extract_pools(document):
    if isinstance(document, list):
        return [p for item in document for p in extract_pools(item)]
    if not isinstance(document, dict):
        return []
    if 'candidates' in document:
        return [document]
    search = document.get('phrase_search') or {}
    context = {key: document[key] for key in ('audio', 'audio_sha256', 'audio_sha', 'seed', 'generation_settings', 'checkpoint_info') if key in document}
    return [{**context, **pool} for pool in extract_pools(search.get('trials', []))]


def audit(documents):
    if isinstance(documents, dict):
        documents = [documents]
    unique, identities, duplicates, conflicts, missing_identity = [], {}, 0, 0, 0
    for document in documents:
        for pool in extract_pools(document):
            identity = (pool.get('audio_sha256'), (pool.get('checkpoint_info') or {}).get('sha256'),
                (pool.get('generation_settings') or {}).get('seed'), pool.get('pool_index'),
                pool.get('mode'), pool.get('start_beat'), pool.get('end_beat'))
            if any(value is None for value in identity):
                missing_identity += 1
            else:
                fingerprint = json.dumps({key: value for key, value in pool.items() if key != 'schema'}, sort_keys=True, ensure_ascii=False)
                previous = identities.setdefault(identity, [])
                if fingerprint in previous:
                    duplicates += 1
                    continue
                if previous:
                    conflicts += 1
                previous.append(fingerprint)
            unique.append(pool)
    runs = {}
    def pool_order(pool):
        run = (pool.get('audio_sha256'), (pool.get('checkpoint_info') or {}).get('sha256'), (pool.get('generation_settings') or {}).get('seed'))
        group = runs.setdefault(run, len(runs))
        index = number(pool.get('pool_index'))
        return group, index if index is not None else float('inf')
    unique.sort(key=pool_order)
    pools = [audit_pool(pool) for pool in unique]
    groups = {}
    for mode in sorted({p['mode'] for p in pools}):
        group = [p for p in pools if p['mode'] == mode]
        keys = group[0]['rank_correlations']
        groups[mode] = dict(pools=len(group), weighting='per_pool_equal',
            mean_rank_correlations={score: {key: mean([p['rank_correlations'][score][key] for p in group])
                for key in ('density', 'events', 'tokens')} for score in keys},
            selected_minus_other_mean={key: mean([(p['selected_minus_other_mean'] or {}).get(key) for p in group])
                for key in ('atoms', 'events', 'tokens', 'comfort', 'timing_support')})
    return dict(schema='composer-v2-candidate-audit-v1', status='available' if pools else 'unavailable',
        unavailable_reason=None if pools else 'No candidate pools: empty input or legacy report without evidence',
        duplicate_pools_removed=duplicates, conflicting_pool_identities=conflicts, pools_without_dedup_identity=missing_identity,
        duplicate_policy='audio sha256/checkpoint sha256/seed/pool_index/mode/beat range; conflicting evidence retained; missing identity cannot deduplicate',
        pools=pools, aggregates=groups,
        notes=['同池每拍分母固定，raw timing sum 与 per-beat 排序相同；这不证明公平。',
               'layout 固定事件结构，动作 token 数仍可能不同。',
               'raw 为模型配置输出、尚未应用 grammar/retry/temperature；grammar 仅含 head 唯一性与语法，不含物理重试。',
               'proposal 为实际 retry/temperature/top-p 提议分布；强制 token 的 proposal logprob 为 0。',
               '跨 stop 事件记录在 boundary_probability，EOS 在 termination_probability；均不纳入这里的范围评分。',
               '重复输入按音频/检查点 SHA、seed、pool_index、mode及拍范围去重；身份相同但证据冲突保留并提示；旧数据缺身份无法去重。',
               '概率仅描述 stop 边界前已评分范围，不声称完整联合似然或 MAP；grammar 每 token 均值仅供比较。',
               '相关性为描述统计，不说明因果；按候选池等权，缺失数据为 unavailable/null。',
               '保留 normal/mix 与休止、重复的自由；本报告不自动判断谱面好坏或生成训练标签。'])


def markdown(report):
    return '\n'.join(['# 同候选池选优审计', f"候选池：{len(report['pools'])}；状态：{report['status']}；移除重复：{report['duplicate_pools_removed']}", *report['notes'],
        *[f"- {mode}: {group['pools']} 池，按池等权。" for mode, group in report['aggregates'].items()]]) + '\n'


def read_input(path):
    if not Path(path).is_file():
        return []
    text = Path(path).read_text(encoding='utf-8-sig').strip()
    if not text:
        return []
    if Path(path).suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--markdown', help='Optional short Markdown report path')
    args = parser.parse_args(argv)
    targets=[Path(args.output).resolve(),*([Path(args.markdown).resolve()] if args.markdown else [])]
    if len(set(targets))!=len(targets) or any(Path(path).resolve() in targets for path in args.input):
        parser.error('output reports must not overwrite inputs or each other')
    documents = [read_input(path) for path in args.input]
    report = audit(documents)
    report['inputs'] = [dict(path=str(Path(path)), status='available' if extract_pools(document) else 'unavailable',
        reason=None if extract_pools(document) else ('missing file' if not Path(path).is_file() else 'empty or legacy file without candidate evidence'))
        for path, document in zip(args.input, documents)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown(report), encoding='utf-8')


if __name__ == '__main__':
    main()
