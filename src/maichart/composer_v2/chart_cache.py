"""Reuse validated exact events and native sustain markers across training runs."""

from fractions import Fraction
import json
from pathlib import Path
import tempfile
import zlib

from maichart.event_timing import EventChart
from maichart.generation.muri.data import REFERENCE_COMMIT
from maichart.generation.sustain_state import SustainStateIndex


# Bump when source parsing, syntax acceptance or persisted marker semantics change.
CHART_CACHE_SCHEMA = 'v1'


class ChartCache:
    def __init__(self, cache_dir, dialect_policy=None):
        self.root = Path(cache_dir) / 'parsed_charts' / CHART_CACHE_SCHEMA / REFERENCE_COMMIT
        if dialect_policy is not None:
            self.root /= dialect_policy

    def _path(self, source_hash, slot):
        return self.root / source_hash[:2] / f'{source_hash}_{slot}.json.zlib'

    def load(self, source_hash, slot):
        path = self._path(source_hash, slot)
        try:
            data = json.loads(zlib.decompress(path.read_bytes()))
            if data['source_hash'] != source_hash or data['chart']['difficulty'] != slot:
                return None
            chart = EventChart.from_dict(data['chart'])
            markers = data['markers']
            sustain = None
            if markers is not None:
                for marker in markers:
                    marker['source_beat'] = Fraction(marker['source_beat'])
                sustain = SustainStateIndex.from_markers(chart.points, chart.tempo_changes,
                                                        chart.audio_first, markers)
            return chart, sustain, data.get('dialect_edits', {})
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, zlib.error):
            return None

    def save(self, source_hash, slot, chart, sustain, dialect_edits):
        path = self._path(source_hash, slot)
        markers = None if sustain is None else [
            {**marker, 'source_beat': str(marker['source_beat'])} for marker in sustain.markers
        ]
        data = dict(source_hash=source_hash, chart=chart.to_dict(), markers=markers,
                    dialect_edits=dialect_edits)
        packed = zlib.compress(json.dumps(data, separators=(',', ':')).encode('utf-8'), level=1)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(packed)
            temporary.replace(path)
        except OSError:
            # Cache availability does not decide whether a source chart is usable.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
