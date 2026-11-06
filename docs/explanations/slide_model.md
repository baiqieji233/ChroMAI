# Slide Model

V1 treats a maimai slide as a chart object that references predefined maimai
trajectory patterns. A slide is not an arbitrary generated path or free-form
geometry.

## Definition

A slide token is modeled as:

```text
start position + predefined trajectory pattern + path arguments + duration
```

Examples:

- `1-4[8:1]`: start at `1`, use pattern `-`, path arg `4`, duration `[8:1]`.
- `2V84[8:1]`: start at `2`, use pattern `V`, path args `8` and `4`,
  duration `[8:1]`.
- `3b-1[8:1]*-7[8:1]`: one break slide head with two parallel branches.

## Token Parts

- `start_position`: the button where the slide starts.
- `pattern`: a known maimai slide pattern such as `-`, `<`, `>`, `^`, `p`, `q`,
  `pp`, `qq`, `s`, `z`, `v`, `V`, or `w`.
- `path_args`: button numbers consumed by the pattern. Most V1 patterns use
  one path arg. `V` uses two.
- `duration`: a raw-preserving duration expression parsed by the unified
  duration parser.
- `segments`: same-head slide branches separated by top-level `*`. Each
  branch keeps its own duration; the object's aggregate duration is the
  longest branch, because the branches are parallel.
- `raw`: original source text.

## ChartIR Shape

V1 stores slide details in `Note.segments`. Each segment has a `trajectory`
object:

```json
{
  "trajectory_id": "1-4",
  "pattern": "-",
  "start_position": 1,
  "end_position": 4,
  "path_args": [4],
  "raw": "1-4"
}
```

For `2V84[8:1]`, the trajectory is:

```json
{
  "trajectory_id": "2V84",
  "pattern": "V",
  "start_position": 2,
  "end_position": 4,
  "path_args": [8, 4],
  "raw": "2V84"
}
```

The final endpoint is retained for ordering and chain continuation, but the
semantics are the selected `V` trajectory plus both path args, not a free line
from `2` to `4`.

## Basic Slides And Same-Head Branches

A basic slide has one segment. A top-level `*` keeps the same slide head and
adds another branch from that head; for example,
`4^7[8:1]*^1[8:1]` means `4^7[8:1]` and `4^1[8:1]`, both starting at `4`.
The endpoint of the first branch is not the start of the second branch.

This is different from `/`: `4-6[8:1]*-7[8:1]` is one same-head slide
object, whereas `4-6[8:1]/4-7[8:1]` is two independent slide heads at the
same timing point. The latter is rejected during validation as a duplicate
slide head because it is not a playable simultaneous pattern.

The `^` pattern is the shortest-arc shorthand. For `4^7`, the parser compares
`4<7` and `4>7` on the eight-button ring and stores the shorter canonical
pattern (`<` or `>`). The symbols depend on the start button: `>` means
clockwise from 1/2/7/8 and counter-clockwise from 3/4/5/6; `<` is the reverse.
Thus `4^7` normalizes to `4<7`, and `4^1` normalizes to `4>1`.
Identical or opposite endpoints have no unique short arc and require an
explicit `<` or `>`; ambiguous `^` paths are validation errors.
The original raw Simai text remains available.

Festival-era and other complex slide syntax should be treated as combinations
or chains of known patterns when V1 can identify every piece. V1 preserves the
raw text and `path_parts` for these compound segments; it does not invent new
geometry.

## V1 Supported Patterns

| Pattern | Category | Path args | Supported | Meaning |
| --- | --- | ---: | --- | --- |
| `-` | line | 1 | yes | Straight slide to one button |
| `<` | arc | 1 | yes | CCW from 1/2/7/8; CW from 3/4/5/6 |
| `>` | arc | 1 | yes | CW from 1/2/7/8; CCW from 3/4/5/6 |
| `^` | shortest-arc | 1 | yes | Choose the shorter arc with the start-button convention |
| `p` | curve | 1 | yes | Known p-shaped maimai slide |
| `q` | curve | 1 | yes | Known q-shaped maimai slide |
| `pp` | curve | 1 | yes | Known pp-shaped maimai slide |
| `qq` | curve | 1 | yes | Known qq-shaped maimai slide |
| `s` | zigzag | 1 | yes | Known s-shaped maimai slide |
| `z` | zigzag | 1 | yes | Known z-shaped maimai slide |
| `v` | composite | 1 | yes | Known lowercase-v slide |
| `V` | composite | 2 | yes | V-shaped slide with two path args |
| `w` | composite | 1 | yes | Known w-shaped maimai slide |

The source of truth for this table is `src/maichart/slides.py`.

The shorthand and shared-head semantics were checked against
[MajSimai's parser](https://github.com/TeamMajdata/MajSimai/blob/master/Runtime/SimaiNoteParser.cs)
and [MaiConverter's arc conversion](https://github.com/donmai-me/MaiConverter/blob/master/maiconverter/simai/simainote.py).

## Not Supported In V1

V1 does not compute real screen coordinates, visual path samples, collision
geometry, slide animation timing, or normalized generated slide syntax.

Future generators must choose from a slide trajectory catalog. If visual path
sampling is needed, samples should come from a predefined trajectory table, not
from a model freely drawing arbitrary curves.
