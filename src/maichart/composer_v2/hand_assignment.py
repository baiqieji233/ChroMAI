"""Whole-group two-hand search over the DX panel.

Edges assign all simultaneous contacts together. Forward and backward costs on
the retained beam expose alternative complete paths through a disputed contact.
These costs describe a heuristic, not measured player probabilities.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

from maichart.composer.stream_quality import _job_position
from maichart.generation.muri.model import CENTERS


_PAD = {tuple(float(v) / 205 for v in xy): key for key, xy in CENTERS.items()}
_MOVE = {"slide", "wifi"}
_EPS = 1.0e-7
EVIDENCE_SCALE = 0.35


@dataclass(frozen=True, slots=True)
class Contact:
    xy: tuple
    kind: str
    phase: str
    job: int
    lane: int = 0
    pad: str = ""
    star_head: bool = False
    fan: int | None = None

    def record(self):
        return dict(
            x=self.xy[0],
            y=self.xy[1],
            kind=self.kind,
            phase=self.phase,
            lane=self.lane,
            pad=self.pad,
            job=self.job,
            star_head=self.star_head,
        )


def _pulse_groups(pulses, heads):
    """Complete-link nearby touches share a palm; retain all sensor identities."""
    groups = []
    for second, xy, kind, radius in sorted(pulses, key=lambda p: (p[2], p[1])):
        pad = _PAD.get(tuple(xy), "")
        lane = int(pad[1:]) if kind == "button" and pad.startswith("A") else 0
        hit = Contact(xy, kind, "pulse", -1, lane, pad, (round(second, 7), xy) in heads)
        if kind == "touch":
            for group in groups:
                if all(
                    a.kind == "touch" and math.dist(xy, a.xy) <= min(0.5, radius + r + 0.15) for a, r in group
                ):
                    group.append((hit, radius))
                    break
            else:
                groups.append([(hit, radius)])
        else:
            groups.append([(hit, radius)])
    return [tuple(c for c, _ in g) for g in groups]


def _logical_jobs(jobs):
    # The two native outer WiFi paths share a fan but require separate hands.
    groups = defaultdict(list)
    for job in jobs:
        # Canonicalize same-time paths independently of Simai '/' ordering.
        segments = tuple((s, t, len(p), p[-1]["x"], p[-1]["y"]) for s, t, p in job["segments"])
        key = (
            job["trace"],
            job["end"],
            job["kind"],
            job["position"],
            job.get("onset", job["trace"]),
            job.get("track_key", ""),
            () if job["kind"] == "wifi" else segments,
        )
        groups[key].append(job)
    return [groups[k] for k in sorted(groups)]


def _timeline(jobs, pulses, warm, horizon, heads, start, end):
    paths = _logical_jobs(jobs)
    times = {round(x[0], 7) for x in pulses}
    # A review may begin within a long Hold or Slide. Emit its hand at both
    # visible boundaries even if the whole excerpt has no new note head.
    times.update((round(start, 7), round(end, 7)))
    for path in paths:
        job = path[0]
        begin, stop = max(warm, job["trace"]), min(horizon, job["end"])
        if begin > stop:
            continue
        times.update((round(begin, 7), round(stop, 7)))
        if job["kind"] in _MOVE:
            # Native path endpoints carry a short release tolerance; keep
            # physical path timing separate from the padded scoring interval.
            for a, b, _ in job["segments"]:
                times.update(round(t, 7) for t in (a, b) if begin <= t <= stop)
            n = min(96, max(1, math.ceil((stop - begin) / 0.05)))
            times.update(round(begin + (stop - begin) * i / n, 7) for i in range(1, n))
    times = sorted(times)
    groups = [[] for _ in times]
    by_time = defaultdict(list)
    for pulse in pulses:
        by_time[round(pulse[0], 7)].append(pulse)
    touch_groups = 0
    for i, t in enumerate(times):
        groups[i] = _pulse_groups(by_time[t], heads)
        touch_groups += sum(g[0].kind == "touch" for g in groups[i])
    job_id = 0
    for fan_id, path in enumerate(paths):
        is_fan = path[0]["kind"] == "wifi"
        branches = path if is_fan else path[:1]
        for job in branches:
            # A release marks the written endpoint before the guard tolerance expires.
            stop = job["segments"][-1][1] if job["segments"] else job["end"]
            for i in range(bisect_left(times, job["trace"] - _EPS), bisect_right(times, stop + _EPS)):
                t = times[i]
                phase = (
                    "start" if abs(t - job["trace"]) <= _EPS else "release" if abs(t - stop) <= _EPS else "active"
                )
                pad = _PAD.get(job["position"], "")
                lane = int(pad[1:]) if pad.startswith("A") else 0
                groups[i].append((Contact(_job_position(job,t),job["kind"],phase,job_id,lane,pad,
                                          fan=fan_id if is_fan else None),))
            job_id += 1
    frames = [(t, tuple(g)) for t, g in zip(times, groups) if g]
    return frames, touch_groups, len(paths)


def _sweeps(frames):
    """Transition support for runs of at least three neighboring rim heads."""
    sweep, members = set(), set()
    last, previous = None, None
    for i, (time, groups) in enumerate(frames):
        strikes = [g[0] for g in groups if g[0].phase in ("start", "pulse")]
        if not strikes:
            continue
        lane = strikes[0].lane if len(strikes) == 1 and strikes[0].kind == "button" else 0
        if last and last[1] and lane:
            dt = time - last[2]
            step = (lane - last[1]) % 8
            if step in (1, 7) and 0.018 <= dt <= 0.24:
                if previous and step == previous[1] and 0.45 <= dt / previous[2] <= 2.2:
                    sweep.update((previous[0], i))
                    members.update((previous[3],previous[0],i))
                previous = (i, step, dt, last[0])
            else:
                previous = None
        else:
            previous = None
        last = (i, lane, time)
    return sweep, members


def _masks(xy):
    n = len(xy)
    if n <= 5:
        return range(1 << n)
    # Bounded alternatives for very dense fan/touch groups. Preserve real
    # sensor identities while trying splits along several physical directions.
    out = {0, (1 << n) - 1}
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        mask = 0
        for i in sorted(range(n), key=lambda i: dx * xy[i][0] + dy * xy[i][1]):
            mask |= 1 << i
            out.update((mask, ((1 << n) - 1) ^ mask))
    return sorted(out)


def _placements(groups, hard_owners):
    contacts = [g[0] for g in groups]
    xy = [tuple(sum(c.xy[k] for c in g) / len(g) for k in (0, 1)) for g in groups]
    single = [i for i, c in enumerate(contacts) if c.phase in ("pulse", "start")]
    button = single[0] if len(single) == 1 and contacts[single[0]].kind == "button" else None
    fans = defaultdict(list)
    for i, contact in enumerate(contacts):
        if contact.fan is not None:
            fans[contact.fan].append(i)
    wings = [pair for pair in fans.values() if len(pair) == 2]
    masks = _masks(xy)
    if len(groups) > 5:
        # Geometric splits alone can omit every continuation of an unusual
        # Hold layout. Always retain a legal extension of every live ownership.
        masks = set(masks)
        geometric = sum((pos[0] >= 0.0) << i for i, pos in enumerate(xy))
        for owners in hard_owners:
            for orientation in (0, 1) if wings else (0,):
                mask = geometric
                for first, second in wings:
                    hand = owners.get(contacts[first].job,orientation)
                    mask = (mask & ~(1 << first | 1 << second)) | (hand << first) | ((1-hand) << second)
                for i, contact in enumerate(contacts):
                    if contact.kind != "slide" and contact.job in owners:
                        mask = (mask & ~(1 << i)) | (owners[contact.job] << i)
                masks.add(mask)
        masks = sorted(masks)
    for mask in masks:
        if any((mask >> a & 1)==(mask >> b & 1) for a,b in wings):
            continue
        reach = 0.0
        targets = []
        owners = []
        for h in (0, 1):
            ix = [i for i in range(len(groups)) if (mask >> i) & 1 == h]
            if not ix:
                targets.append(None)
                continue
            center = tuple(sum(xy[i][k] for i in ix) / len(ix) for k in (0, 1))
            span = max((math.dist(xy[i], xy[j]) for k, i in enumerate(ix) for j in ix[:k]), default=0.0)
            # Simultaneous independent rim taps favor two hands. Small palm
            # spans remain available when the other hand has an ongoing task.
            pulse_count = sum(contacts[i].kind == "button" for i in ix)
            reach += max(0.0, span - 0.42) * 1.9 + max(0, pulse_count - 1) * 2.2
            releases = [i for i in ix if contacts[i].kind=="hold" and contacts[i].phase=="release"]
            # A hand still at a Hold's release does not have a full movement
            # interval to reach a different button at that same instant.
            for i in releases:
                for j in ix:
                    if contacts[j].kind == "button":
                        reach += 2.8 * max(0.,math.dist(xy[i],xy[j])-.42)
            js = tuple(contacts[i].job for i in ix if contacts[i].job >= 0)
            sustain_only = bool(js) and len(js) == len(ix)
            lane = contacts[ix[0]].lane if len(ix) == 1 and contacts[ix[0]].kind == "button" else 0
            targets.append((center, js, sustain_only, lane))
            owners.extend((j, h) for j in js)
        yield (
            mask,
            targets,
            tuple(sorted(owners)),
            reach,
            ((mask >> button) & 1 if button is not None else -1),
        )


@dataclass(slots=True)
class Node:
    hands: tuple
    owners: tuple
    actor: int
    strike: float
    cost: float
    parent: int
    mask: int
    edges: list


def _advance(node, old_owners, placement, time, sweep, rim_run, has_strike,
             held_by):
    mask, targets, owners, cost, actor = placement
    hands = list(node.hands)
    for h, target in enumerate(targets):
        if target is None:
            continue
        xy, jobs, sustain_only, lane = target
        prev, t, old_lane, old_step, old_dt = node.hands[h]
        elapsed = max(0.035, time - t)
        step = (lane - old_lane + 4) % 8 - 4 if lane and old_lane else 0
        continuing = sustain_only and all(old_owners.get(j) == h for j in jobs)
        if not continuing:
            distance = math.dist(xy, prev)
            # A sweep has continuous tangential travel across ring boundaries.
            # Scaling its travel cost only on the hand which actually continues
            # leaves an occupied or poorly placed hand free to transfer.
            speed = distance / elapsed
            if sweep and actor == h and node.actor == h:
                speed *= 0.31
            elif (
                step
                and abs(step) <= 2
                and old_step == step
                and 0.5 * old_dt <= elapsed <= 1.8 * old_dt
                and elapsed <= 0.4
            ):
                # Either hand can maintain an established angular movement
                # during successive simultaneous pairs, even with a different
                # stride on the other hand. Sudden jumps get no such discount.
                speed *= 0.38
            cost += min(4.0, max(0.0, speed - 3.1) / 6)
            if has_strike:
                opposite_side = max(0.0, (xy[0] if h == 0 else -xy[0]) - 0.39)
                # Preserve long circumferential movements. In short independent
                # rim hits, reaching deep into the opposite half needs a reason.
                reach = 0.65 if lane and old_lane and lane != old_lane and not rim_run else 0.09
                cost += reach * opposite_side
            if (actor == h and node.actor == h and
                    old_lane and lane != old_lane and not sweep and 1-h not in held_by):
                cost += 0.36 * max(0.0, min(1.0, (0.20 - elapsed) / 0.11))
        hands[h] = (xy, time, lane, step, elapsed if lane else 0.0)
    if has_strike and all(time - hand[1] < 0.28 for hand in hands):
        cost += 0.12 * max(0.0,hands[0][0][0]-hands[1][0][0]-.28)
    if sweep and actor >= 0 and node.actor >= 0 and actor != node.actor:
        cost += 0.27
    return tuple(hands), cost


def search(frames, warm, beam_width, record):
    sweep, rim_run = _sweeps(frames)
    initial = Node(
        (((-0.55, 0.0), warm - 1, 0, 0, 0.0), ((0.55, 0.0), warm - 1, 0, 0, 0.0)),
        (),
        -1,
        -math.inf,
        0.0,
        0,
        0,
        [],
    )
    beam = [initial]
    layers = [beam]
    for i, (time, groups) in enumerate(frames):
        has_strike = any(g[0].phase in ("pulse", "start") for g in groups)
        places = tuple(_placements(groups, (dict(node.owners) for node in beam)))
        job_kind = {g[0].job: g[0].kind for g in groups if g[0].job >= 0}
        active_holds = {
            group[0].job for group in groups
            if group[0].kind in ("hold", "touch_hold") and group[0].phase != "release"
        }
        selected = {}
        for parent, node in enumerate(beam):
            old = dict(node.owners)
            for place in places:
                mask, _, owners, _, actor = place
                transfers = [j for j, h in owners if j in old and old[j] != h]
                if any(job_kind[j] != "slide" for j in transfers):
                    continue
                held_by = {h for j,h in owners if j in active_holds} if active_holds else ()
                hands, increment = _advance(node,old,place,time,i in sweep,i in rim_run,has_strike,held_by)
                increment += 0.26 * len(transfers)
                total = node.cost + increment
                a = actor if has_strike else node.actor
                t = time if has_strike else node.strike
                key = (hands, owners, a, t)
                old_node = selected.get(key)
                if old_node is None:
                    edges = [(parent, mask, increment)] if record else []
                    selected[key] = Node(hands, owners, a, t, total, parent, mask, edges)
                else:
                    if record:
                        old_node.edges.append((parent, mask, increment))
                    if total < old_node.cost:
                        old_node.cost, old_node.parent, old_node.mask = total, parent, mask
        beam = sorted(selected.values(), key=lambda n: n.cost)[:beam_width]
        if not beam:
            raise RuntimeError("No valid hand search state")
        layers.append(beam)
    return layers


def _best_masks(layers):
    best = []
    k = 0
    for layer in reversed(layers[1:]):
        node = layer[k]
        best.append(node.mask)
        k = node.parent
    return best[::-1]


def _two_sided_evidence(costs, first, second):
    """Normalize the two best constrained complete paths, without counting clones."""
    support = [name for name, cost in zip((first, second), costs) if math.isfinite(cost)]
    if len(support) != 2:
        return {first: None, second: None, "seen_both": False, "support": support, "cost_gap": None}
    low = min(costs)
    a, b = (math.exp(-(c - low) / EVIDENCE_SCALE) for c in costs)
    return {
        first: a / (a + b),
        second: b / (a + b),
        "seen_both": True,
        "support": support,
        "cost_gap": abs(costs[0] - costs[1]),
    }


def _paths(layers, frames, start, end, count):
    # Each retained edge has a minimum total complete-path cost. This also
    # retains alternatives that merge back into the best state before the end.
    n = len(frames)
    beam = len(layers[-1])
    suffix = [0.0] * beam
    next_edges = [None] * n
    rivals = [None] * n
    different_motion = [None] * n
    best = _best_masks(layers)
    relations = []
    previous = -1
    last_sec = -math.inf
    contact_evidence = [None] * n
    relation_evidence = [None] * n
    for (time, groups), mask in zip(frames, best):
        strikes = [j for j, g in enumerate(groups) if g[0].phase in ("pulse", "start")]
        actor = (
            (mask >> strikes[0]) & 1 if len(strikes) == 1 and groups[strikes[0]][0].kind == "button" else -1
        )
        relations.append(
            actor != previous if min(actor, previous) >= 0 and 0.02 < time - last_sec <= 0.45 else None
        )
        if strikes:
            previous, last_sec = actor, time
    for i in reversed(range(n)):
        prev = layers[i]
        costs = [math.inf] * len(prev)
        edges = [None] * len(prev)
        by_mask = {}
        relation_costs = [math.inf, math.inf]
        for k, node in enumerate(layers[i + 1]):
            if not math.isfinite(suffix[k]):
                continue
            for parent, mask, increment in node.edges:
                remaining = increment + suffix[k]
                if remaining < costs[parent]:
                    costs[parent] = remaining
                    edges[parent] = (k, mask)
                candidate = (prev[parent].cost + remaining, parent, mask, k)
                by_mask[mask] = min(by_mask.get(mask, math.inf), candidate[0])
                if mask != best[i]:
                    if rivals[i] is None or candidate < rivals[i]:
                        rivals[i] = candidate
                if relations[i] is not None and min(node.actor, prev[parent].actor) >= 0:
                    changed = node.actor != prev[parent].actor
                    relation_costs[changed] = min(relation_costs[changed], candidate[0])
                    if changed != relations[i] and (
                        different_motion[i] is None or candidate < different_motion[i]
                    ):
                        different_motion[i] = candidate
        group_costs = [[math.inf, math.inf] for _ in frames[i][1]]
        for mask, value in by_mask.items():
            for j, sides in enumerate(group_costs):
                h = (mask >> j) & 1
                sides[h] = min(sides[h], value)
        contact_evidence[i] = [_two_sided_evidence(sides, "left", "right") for sides in group_costs]
        if relations[i] is not None:
            relation_evidence[i] = _two_sided_evidence(relation_costs, "same", "change")
        next_edges[i] = edges
        suffix = costs
    total = layers[-1][0].cost
    results = [(total, best)]
    order = sorted(
        (
            (i, edge, kind)
            for kind, routes in enumerate((different_motion, rivals))
            for i, edge in enumerate(routes)
            if edge and edge[0] - total <= 0.8 and start <= frames[i][0] <= end
        ),
        key=lambda item: (
            item[2],
            round(item[1][0] - total, 6),
            abs(frames[item[0]][0] - (start + end) / 2),
            item[0],
        ),
    )
    for i, edge, _ in order:
        if len(results) >= count:
            break
        cost, parent, mask, k = edge
        if any(
            abs(cost - other) < 1e-8
            and route[i] == mask
            and (i == 0 or route[i - 1] == layers[i][parent].mask)
            for other, route in results
        ):
            continue
        route = [0] * n
        route[i] = mask
        for j in reversed(range(i)):
            node = layers[j + 1][parent]
            route[j] = node.mask
            parent = node.parent
        for j in range(i + 1, n):
            k, route[j] = next_edges[j][k]
        if all(route != prev for _, prev in results):
            results.append((cost, route))

    def gaps(items):
        return [None if r is None else max(0.0, r[0] - total) for r in items]

    return results, gaps(rivals), gaps(different_motion), contact_evidence, relation_evidence


def analyze(stream, start, end, observed_end, beam_width, max_solutions, export_frames):
    if not all(math.isfinite(v) for v in (start, end, observed_end)) or end < start:
        raise ValueError("Hand recognition needs finite seconds and end >= start")
    if not (1 <= beam_width <= 128 and 1 <= max_solutions <= 16):
        raise ValueError("Hand recognition beam must be 1..128 and solutions 1..16")
    warm = start - 0.6
    horizon = max(end, min(end + 0.35, observed_end))
    take = (stream.job_ends >= warm) & (stream.job_starts <= horizon)
    jobs = [stream.jobs[i] for i in np.flatnonzero(take)]
    a = stream.pulse_seconds.searchsorted(warm, side="left")
    b = stream.pulse_seconds.searchsorted(horizon, side="right")
    frames, touches, njobs = _timeline(jobs, stream.pulses[a:b], warm, horizon, stream.star_heads, start, end)
    layers = search(frames, warm, beam_width, export_frames)
    if export_frames:
        routes, gaps, motion_gaps, contact_evidence, relation_evidence = _paths(
            layers, frames, start, end, max_solutions
        )
    else:
        routes = [(layers[-1][0].cost, _best_masks(layers))]
    sols = []
    stats = None
    for cost, masks in routes:
        hits = pairs = 0
        last = -1
        last_t = -math.inf
        output = []
        for i, ((second, groups), mask) in enumerate(zip(frames, masks)):
            strike = [j for j, g in enumerate(groups) if g[0].phase in ("pulse", "start")]
            if strike:
                actor = (
                    (mask >> strike[0]) & 1
                    if len(strike) == 1 and groups[strike[0]][0].kind == "button"
                    else -1
                )
                if min(last, actor) >= 0 and start <= second <= end and 0.02 < second - last_t <= 0.45:
                    pairs += 1
                    hits += last != actor
                last, last_t = actor, second
            if export_frames and start <= second <= end:
                left = []
                right = []
                for j, g in enumerate(groups):
                    evidence = contact_evidence[i][j]
                    contacts = right if (mask >> j) & 1 else left
                    for contact in g:
                        contacts.append(
                            dict(
                                contact.record(),
                                p_left=evidence["left"],
                                p_right=evidence["right"],
                                seen_both=evidence["seen_both"],
                                hand_support=evidence["support"],
                                hand_cost_gap=evidence["cost_gap"],
                            )
                        )
                output.append(
                    dict(
                        second=second,
                        left=left,
                        right=right,
                        alternate_cost_gap=gaps[i],
                        relation_cost_gap=motion_gaps[i],
                        relation_evidence=relation_evidence[i],
                    )
                )
        sols.append(dict(cost=cost, frames=output))
        if stats is None:
            stats = (hits, pairs)
    # Reconstruct scored edge increments of the optimal path, excluding warmup.
    k = 0
    selected_cost = 0.0
    max_step = 0.0
    hemi = 0.0
    scored = 0
    for i in reversed(range(len(frames))):
        node = layers[i + 1][k]
        parent = layers[i][node.parent]
        if frames[i][0] >= start:
            increment = node.cost - parent.cost
            selected_cost += increment
            max_step = max(max_step, increment)
            scored += 1
            for j, g in enumerate(frames[i][1]):
                if g[0].phase in ("pulse", "start"):
                    sign = -1 if (node.mask >> j) & 1 else 1
                    hemi += 0.09 * max(0.0, sign * g[0].xy[0] - 0.39)
        k = node.parent
    unknown = stream.coverage["unknown"]
    actions = njobs + b - a
    known = actions / max(1, actions + len(unknown)) if unknown else 1.0
    score = math.exp(-selected_cost / max(1.0, horizon - start))
    coverage = dict(
        stream.coverage,
        unknown=unknown[:32],
        known_fraction=float(known),
        unknown_count=len(unknown),
        complete=not unknown,
        active_jobs=njobs,
        touch_contact_groups=touches,
        sample_step_sec=0.05,
        max_samples_per_action=96,
        lookahead_sec=horizon - end,
        warmup_sec=0.6,
        events=scored,
        best_cost=selected_cost,
        max_step_penalty=max_step,
        hand_assignment="group_path_v1",
        beam_width=beam_width,
        alternation_pairs=stats[1],
        alternation_ratio=stats[0] / stats[1] if stats[1] else 0.0,
        mean_hemi_penalty=hemi / max(1, scored),
    )
    return dict(
        score=float(known * score + (1 - known) * 0.5),
        coverage=coverage,
        solutions=sols,
        hand_evidence=dict(
            method="two_min_complete_path_costs_on_retained_beam",
            scale=EVIDENCE_SCALE,
            calibrated=False,
            single_sided="unknown_probability",
        ),
    )
