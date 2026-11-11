"""Per-kind playability policy shared by static and MiaCode checks.

The policy changes which findings enter the repair/acceptance stream.  It does
not reinterpret the native alert level: a reported native ``Muri`` remains a
Muri finding even when the configured action is ``report``.
"""
from __future__ import annotations

from collections.abc import Mapping


CONFLICT_KINDS = (
    "MultiTouch",
    "TouchMultiAction",
    "Overlap",
    "TapOnSlide",
    "SlideHeadTap",
    "SlideTooFast",
)
CONFLICT_ACTIONS = ("off", "report", "enforce")


def validate_kind_modes(modes):
    """Validate and normalize ``conflict_kind_modes``.

    ``None`` is deliberately preserved as the legacy policy (all detected kinds
    enforce). A mapping may omit kinds; omitted entries also default to
    ``enforce`` so adding a new call site cannot silently disable a check.
    """
    if modes is None:
        return None
    if not isinstance(modes, Mapping):
        raise ValueError("conflict_kind_modes must be a mapping or None")
    unknown = set(modes) - set(CONFLICT_KINDS)
    if unknown:
        raise ValueError("unknown conflict kinds: " + ", ".join(sorted(map(str, unknown))))
    normalized = {}
    for kind, action in modes.items():
        if kind not in CONFLICT_KINDS:
            raise ValueError(f"unknown conflict kind: {kind}")
        if action not in CONFLICT_ACTIONS:
            raise ValueError(f"invalid action for {kind}: {action!r}")
        normalized[kind] = action
    return normalized


def kind_mode(modes, kind: str) -> str:
    """Return the configured action for a native kind or issue type."""
    normalized = str(kind)
    if normalized.startswith("miacode_"):
        normalized = normalized[len("miacode_"):]
    if modes is None:
        return "enforce"
    return modes.get(normalized, "enforce")


def issue_kind(issue) -> str:
    return str(getattr(issue, "conflict_type", "")).removeprefix("miacode_")


def action_for_issue(issue, modes) -> str:
    return kind_mode(modes, issue_kind(issue))


def is_actionable(issue) -> bool:
    """Whether an issue may trigger repair or rejection."""
    return getattr(issue, "action", "enforce") == "enforce"


def apply_issue_policy(issue, modes):
    """Apply one action and return the issue, or ``None`` for ``off``.

    The caller owns ignored evidence accounting.  Returning ``None`` keeps
    the normal issue stream free of disabled kinds while raw native evidence
    remains available in the coverage payload.
    """
    action = action_for_issue(issue, modes)
    issue.action = action
    if action == "off":
        return None
    if action == "report":
        issue.repair_target = None
    return issue


def ignored_issue_record(issue) -> dict:
    """Serialize an ignored issue for coverage/audit output."""
    issue.action = "off"
    issue.repair_target = None
    return issue.to_dict()


def validate_tap_on_slide_threshold_ms(value) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("tap_on_slide_threshold_ms must be a finite number") from exc
    if threshold != threshold or threshold in (float("inf"), float("-inf")):
        raise ValueError("tap_on_slide_threshold_ms must be finite")
    if not 0.0 <= threshold <= 250.0:
        raise ValueError("tap_on_slide_threshold_ms must be between 0 and 250")
    return threshold

