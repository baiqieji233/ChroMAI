"""Development-only differential check against the pinned ABI3 oracle.

The native library is used only by this explicit developer command. Production
CLI/WebUI imports the Python engine and never imports this module.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = [
    "(120){4}1h[1:1],,,,1,E", "(120){4}1h[1:1],,,,1/2b/C,E",
    "(120){4}1/1/2,E", "(120){4}1/3/6,E", "(120){4}1/5/C,E",
    "(120){4}B1/B2/B3/B4/B5/B6/B7/B8/C,E",
    "(150){4}3>1[4:1],5<1[4:1],1>5[4:1],5<1[4:1],1,E",
    "(120){4}1-5[4:1],,{20},5,E", "(120){4}1w5[4:1],,5/6/4,E",
    "(120){4}1<8-4[4:1],E", "(120){4}8z4[2:1],E",
    "(120){4}1-5[4:1]*-4[4:1],E", "(120){4}1-5[4:1]/1-4[4:1],E",
    "(120){4}1h[1:1],3/6,E", "(120){4}1h[1:1]/5h[1:1],3/7,E",
    "(120){4}1h[1:1],A1,E", "(120){4}Ch[1:1],C,E",
    "(120){4}1-5[2:1],,A1/B1/C/B5/A5,E", "(120){4}1-5[4:1],1,E",
    "(120){4}1-5[4:1],{40},1,E", "(120){4}1-5[4:1],{40},1x,E",
    "(120){4}1-5[4:1],{20},1,E", "(120){4}1-5[4:1],{20},1/2,E",
    "(120){4}1-5[0.2##2],{16}2,3,4,5,6,7,8,E",
    "(120){4}1-5[4:1],5-1[4:1],E", "(120){4}1-5[4:1],(150)5h[4:1],5,E",
    "(120){4}1w5[2:1],{16}3,4,5,6,7,E", "(120){4}1p5[4:1],E",
    "(120){4}1pp5[4:1],E", "(120){4}1V35[4:1],E",
    "(120){4}1-5[4:1]-1[8:1],E", "(120){4}1^3[4:1],E",
    "(120){4}1-5[#0.5],E", "(120){4}1-5[0.2##0.5],1/2,E",
    "(120){4}1-5[180#4:1],E", "(120){4}1-5[0.1##180#4:1],E",
    "(120){4}1h[#0.5],1,E", "(120){4}1h,1,E",
    "(120){4}1h[1:0]/1/1,E", "(120){4}1x/1/2,E",
    "(120){4}1bx-5b[4:1],E", "(120){4}1-5[4:1]*-7[8:1],E",
    "(120){4}1q5[4:1],1qq5[4:1],1s5[4:1],1v5[4:1],E",
    "(120){4}1h[1:1],(240){24}1/2,A1/B1/C,,,,1,E",
    "(120){4}\n1-5[4:1],\n{16}1,1,1,A1,\n5/6,E",
]


def oracle():
    directory = ROOT / ".native/miacode/runtime"
    name = "maichart_muri_v3.dll" if os.name == "nt" else "libmaichart_muri_v3.so"
    from maichart.generation.muri.data import REFERENCE_COMMIT
    provenance = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    if provenance["commit"] != REFERENCE_COMMIT or provenance["abi"] != 3:
        raise RuntimeError("Expected the pinned ABI3 native development oracle")
    for file in (name, "Qt6Core.dll", "Qt6Gui.dll") if os.name == "nt" else (name,):
        if hashlib.sha256((directory / file).read_bytes()).hexdigest() != provenance["files"][file]:
            raise RuntimeError(f"Native oracle checksum mismatch: {file}")
    dll_dir = os.add_dll_directory(str(directory)) if os.name == "nt" else None
    lib = ctypes.CDLL(str(directory / name))
    lib.maichart_muri_abi.restype = ctypes.c_int
    if lib.maichart_muri_abi() != 3:
        raise RuntimeError("Expected ABI3")
    lib.maichart_muri_analyze_v3.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    lib.maichart_muri_analyze_v3.restype = ctypes.c_void_p
    lib.maichart_muri_free.argtypes = [ctypes.c_void_p]
    def analyze(text, first, threshold):
        ptr = lib.maichart_muri_analyze_v3(text.encode(), first, threshold/1000.)
        try:
            return json.loads(ctypes.string_at(ptr))
        finally:
            lib.maichart_muri_free(ptr)
    analyze.dll_dir = dll_dir
    return analyze


def normalized(report):
    markers = {m["key"]: m for m in report["markers"]}
    for ref in report["static_references"]:
        for name in ("affected", "cause"):
            m = ref[name]
            markers[m["marker_key"]] = m
    def signature(d):
        marker = markers.get(d["marker_key"], d)
        return d["kind"], d["alert_level"], marker["line"], marker["col"], round(d["second"], 6)
    return {"ok": report["ok"], "summary": report["summary"],
            "diagnostics": sorted(signature(d) for d in report["diagnostics"]),
            "static": sorted((r["kind"], r["alert_level"], r["affected"]["line"], r["affected"]["col"],
                              r["cause"]["line"], r["cause"]["col"], round(r.get("delta_second", 0), 6))
                             for r in report["static_references"])}


def equivalent(left, right):
    """Native doubles may differ by sub-microseconds after JSON roundtrip."""
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(left-right) <= 0.000002
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    return left == right


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / ".scratch/muri_python_port/parity.json")
    parser.add_argument("--chart", type=Path, action="append", default=[], help="Also compare inote_5 from an existing maidata at 200 ms")
    args = parser.parse_args()
    from maichart.generation.miacode_python import analyze_inote
    native = oracle()
    results = []
    snapshots = []
    for text in FIXTURES:
        for threshold in (0, 100, 200, 250):
            for first in (0., -0.25):
                expected = native(text, first, threshold)
                actual = analyze_inote(text, first=first, tap_on_slide_threshold_ms=threshold)
                left, right = normalized(expected), normalized(actual)
                snapshots.append(dict(inote=text, first=first, threshold_ms=threshold, expected=left))
                matches = equivalent(left, right)
                results.append(dict(inote=text, first=first, threshold_ms=threshold, match=matches,
                                    **({} if matches else dict(expected=left, actual=right))))
    args.report.parent.mkdir(exist_ok=True, parents=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.snapshot:
        args.snapshot.parent.mkdir(exist_ok=True, parents=True)
        args.snapshot.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [r for r in results if not r["match"]]
    chart_results = []
    if args.chart:
        from maichart.maidata import parse_maidata_metadata
        for path in args.chart:
            chart = parse_maidata_metadata(path.read_text(encoding="utf-8-sig"))
            text = chart.fields["inote_5"]
            if chart.wholebpm:
                text = f"({float(chart.wholebpm):.12g})" + text
            first = float(chart.first or 0)
            expected = native(text, first, 200)
            actual = analyze_inote(text, first=first)
            left, right = normalized(expected), normalized(actual)
            match = equivalent(left, right)
            record = dict(path=str(path), match=match, expected=left, actual=right,
                          python_elapsed_sec=actual["elapsed_sec"], note_count=actual["summary"]["note_count"],
                          parse_errors=actual["parse"]["errors"])
            chart_results.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in {"expected", "actual"}}, ensure_ascii=False))
        args.report.with_suffix(".charts.json").write_text(json.dumps(chart_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "matched": len(results)-len(failures), "different": len(failures),
                      "report": str(args.report)}, ensure_ascii=False))
    for fail in failures[:4]:
        print(json.dumps(fail, ensure_ascii=False))
    return bool(failures) or any(not r["match"] for r in chart_results)


if __name__ == "__main__":
    raise SystemExit(main())
