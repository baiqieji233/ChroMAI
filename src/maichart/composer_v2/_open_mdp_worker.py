"""Private CPU worker. Launch from the verified open-mdp inference directory."""
from collections import defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy
import sys


def _features(entries):
    notes_by_time = defaultdict(list)
    for entry in entries:
        notes_by_time[entry['Time']].extend(entry['Notes'])
    times = sorted(notes_by_time)
    density_start = index = 0
    timeline = []
    previous = None
    for time in times:
        group = notes_by_time[time]
        group.sort(key=lambda note: note['startPosition'])
        while density_start < index and timeline[density_start] < time - 1.5:
            density_start += 1
        for note in group:
            position = note['startPosition']
            distance = 0 if previous is None else abs(position - previous)
            note.update(time=time, density=index - density_start, multiPressCount=len(group),
                        sweepAllowed=False, displacement=min(distance, 8 - distance))
            timeline.append(time)
            index += 1
            previous = position

    # The upstream search enumerates all choices in a half-second window.
    # A sweep depends only on three consecutive onset groups within that window.
    positions = []
    for time in times:
        group = defaultdict(list)
        for note in notes_by_time[time]:
            group[note['startPosition'] % 8].append(note)
        positions.append(group)
    for index in range(2, len(times)):
        if times[index - 2] < times[index] - .5:
            continue
        first, middle, last = positions[index - 2:index + 1]
        for lane, notes in first.items():
            for direction in (1, -1):
                second = middle.get((lane + direction) % 8)
                third = last.get((lane + 2 * direction) % 8)
                if second and third:
                    for note in (*notes, *second, *third):
                        note['sweepAllowed'] = True
    for entry in entries:
        entry['Notes'] = notes_by_time[entry['Time']]


def main():
    with redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(1)
        inference = runpy.run_path('src/inference.py')
    print(json.dumps({'ready': True}))
    for line in sys.stdin:
        try:
            file = Path(json.loads(line))
            entries = json.loads(file.read_text(encoding='utf-8'))
            _features(entries)
            file.write_text(json.dumps(entries), encoding='utf-8')
            with redirect_stdout(sys.stderr):
                _, level = inference['predict_difficulty'](file)
            response = {'level': level}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            response = {'error': str(error)}
        print(json.dumps(response))


if __name__ == '__main__':
    main()
