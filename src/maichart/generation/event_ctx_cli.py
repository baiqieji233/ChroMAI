"""Generate with the cross-window family (``event_ctx_v1``).

Same code as ``event_cli`` with the family's defaults filled in and the
checkpoint checked against the family contract first, so a mismatched
checkpoint fails with a clear message instead of producing a chart with an
arbitrary seam.

The flags are identical, and any of them can still be overridden::

    python -m maichart.generation.event_ctx_cli --checkpoint ctx.pt --audio song.mp3 \
        --output out/maidata.txt --context-beats 8 --audio-margin-beats 4
"""

from __future__ import annotations

import sys

from maichart.generation import event_cli
from maichart.generation.event_ctx import (
    DEFAULT_AUDIO_MARGIN_BEATS,
    DEFAULT_CONTEXT_BEATS,
    DEFAULT_CONTEXT_TOKENS,
    TARGET_SCHEMA,
)
from maichart.training.event_checkpoints import load_checkpoint

DEFAULT_FLAGS: tuple[tuple[str, object], ...] = (
    ("--context-beats", DEFAULT_CONTEXT_BEATS),
    ("--audio-margin-beats", DEFAULT_AUDIO_MARGIN_BEATS),
    ("--context-tokens", DEFAULT_CONTEXT_TOKENS),
)


def _with_defaults(argv: list[str], contract: dict | None = None) -> list[str]:
    """Fill in the window layout, preferring what the checkpoint was trained with.

    The layout is part of the data contract: generating with a different lead-in
    length feeds the model something it never saw, which shows up as "a couple of
    notes per window and then <eos>".  So an unset flag takes the checkpoint's own
    value, and a flag the caller set explicitly is honoured but compared.
    """

    result = list(argv)
    trained = (contract or {}).get("family_params") or {}
    keys = {
        "--context-beats": "context_beats",
        "--audio-margin-beats": "audio_margin_beats",
        "--context-tokens": "context_tokens",
        "--overlap-beats": "overlap_beats",
    }
    for flag, value in DEFAULT_FLAGS:
        if flag not in result:
            trained_value = trained.get(keys[flag])
            result.extend([flag, str(trained_value if trained_value is not None else value)])
    if "--overlap-beats" not in result:
        result.extend(["--overlap-beats", str(trained.get("overlap_beats", 0))])
    return result


def _check_layout(argv: list[str], contract: dict) -> None:
    """Warn when the caller's layout differs from the training contract."""

    trained = (contract or {}).get("family_params") or {}
    if not trained:
        return
    for flag, key in (("--context-beats", "context_beats"),
                      ("--audio-margin-beats", "audio_margin_beats"),
                      ("--context-tokens", "context_tokens"),
                      ("--overlap-beats", "overlap_beats")):
        if flag not in argv:
            continue
        provided = argv[argv.index(flag) + 1]
        expected = trained.get(key)
        if expected is not None and str(provided) != str(expected):
            print(f"warning: {flag}={provided} 与训练时不一致（训练用 {expected}）。"
                  f"窗口布局属于数据合同，不一致会让模型收到没见过的输入，"
                  f"典型表现是每个窗口只输出几个音符就结束。", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Fail fast and clearly when the checkpoint belongs to another family: both
    # families share the checkpoint schema, so only the contract tells them
    # apart.
    contract: dict | None = None
    if "--checkpoint" in arguments:
        try:
            path = arguments[arguments.index("--checkpoint") + 1]
            payload = load_checkpoint(path, map_location="cpu")
        except Exception:  # noqa: BLE001 - the real loader reports it properly
            payload = None
        if payload is not None:
            schema = payload.get("data_contract", {}).get("target_schema")
            if schema != TARGET_SCHEMA:
                print(
                    f"checkpoint_family_mismatch expected={TARGET_SCHEMA} found={schema!r}\n"
                    "这个 checkpoint 属于另一个模型族（例如 event_v1）。跨窗口模型必须用它自己训练的权重：\n"
                    f"  python -m maichart.training.train_event_ctx ...   （训练 {TARGET_SCHEMA}）",
                    file=sys.stderr, flush=True,
                )
                return 2
            contract = payload.get("data_contract", {})
    if contract is not None:
        _check_layout(arguments, contract)
    return event_cli.main(_with_defaults(arguments, contract))


if __name__ == "__main__":
    raise SystemExit(main())
