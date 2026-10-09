"""Generate a Simai chart from audio using a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from maichart.generation.conditions import GenerationCondition
from maichart.generation.inference import (
    audio_to_frame_tensor,
    build_tempo_changes_from_audio,
    load_model_checkpoint,
    load_or_analyze_audio,
    predict_frame_notes,
)
from maichart.generation.simai_compile import compile_predictions, parse_tempo_changes, save_generated_ir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    condition = GenerationCondition.from_json(args.condition_file) if args.condition_file else GenerationCondition(level=args.level)
    features = load_or_analyze_audio(args.audio, args.audio_features, division=args.division, sample_rate=args.sample_rate)
    bpm = args.bpm or features.tempo_bpm
    if bpm <= 0:
        raise SystemExit("BPM could not be estimated. Provide --bpm.")
    tempo_changes = parse_tempo_changes(args.tempo_map) if args.tempo_map else None
    if tempo_changes is None and args.bpm is None:
        tempo_changes = build_tempo_changes_from_audio(
            features,
            bpm=bpm,
            first=args.first,
            division=args.division,
        )
        bpm = tempo_changes[0].bpm
    frame_features, frame_count = audio_to_frame_tensor(
        features,
        bpm=bpm,
        division=args.division,
        first=args.first,
        tempo_changes=tempo_changes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for candidate in range(args.candidates):
        torch.manual_seed(args.seed + candidate)
        events = predict_frame_notes(
            model,
            frame_features,
            condition,
            device=device,
            presence_threshold=args.presence_threshold,
            button_threshold=args.button_threshold,
            sample=args.candidates > 1,
            guidance_scale=args.guidance_scale,
        )
        simai, chart = compile_predictions(
            events,
            frame_count=frame_count,
            bpm=bpm,
            division=args.division,
            first=args.first,
            tempo_changes=tempo_changes,
            title=args.title,
            artist=args.artist,
            level=args.chart_level or str(condition.level),
            designer=args.designer,
        )
        target = _candidate_path(output, candidate, args.candidates)
        target.write_text(_maidata(chart, simai), encoding="utf-8")
        if args.output_ir:
            save_generated_ir(chart, _candidate_path(Path(args.output_ir), candidate, args.candidates))
        print(f"candidate={candidate + 1}/{args.candidates} output={target} frames={frame_count} checkpoint_epoch={checkpoint.get('epoch')}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio")
    input_group.add_argument("--audio-features")
    parser.add_argument("--output", required=True, help="maidata.txt path; candidates add a numeric suffix")
    parser.add_argument("--output-ir")
    parser.add_argument("--condition-file")
    parser.add_argument("--level", type=float, default=13.0)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--first", type=float, default=0.0)
    parser.add_argument("--division", type=int, default=16)
    parser.add_argument("--tempo-map")
    parser.add_argument("--presence-threshold", type=float, default=0.45)
    parser.add_argument("--button-threshold", type=float, default=0.45)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="条件引导强度；1 为正常，0 接近无条件，2 以上更强但可能牺牲自然度",
    )
    parser.add_argument("--candidates", type=int, default=1, help="manual opt-in; no ranking or playability filtering is applied")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--title", default="Generated maimai chart")
    parser.add_argument("--artist", default="")
    parser.add_argument("--chart-level", default="")
    parser.add_argument("--designer", default="AI")
    return parser


def _maidata(chart, simai: str) -> str:
    difficulty = chart.difficulty.index or 5
    return "\n".join(
        [
            f"&title={chart.metadata.title or ''}",
            f"&artist={chart.metadata.artist or ''}",
            f"&first={chart.metadata.offset or 0}",
            f"&wholebpm={chart.timing.bpms[0].bpm if chart.timing.bpms else 120}",
            f"&lv_{difficulty}={chart.difficulty.level or ''}",
            f"&des_{difficulty}={chart.difficulty.designer or ''}",
            f"&inote_{difficulty}=",
            simai,
            "",
        ]
    )


def _candidate_path(path: Path, index: int, count: int) -> Path:
    if count == 1:
        return path
    return path.with_name(f"{path.stem}_candidate_{index + 1:03d}{path.suffix}")


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    raise SystemExit(main())
