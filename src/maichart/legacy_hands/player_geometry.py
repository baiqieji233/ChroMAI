"""Physical scale for the maimai DX player preview.

The SEGA operator drawing is a 1:100 cabinet layout, not a tolerance or
sensor map. The nominal dimensions below are estimates from its orthographic
DXF views: the circular opening spans about 525 mm, the button sectors are
centered about 315–320 mm from the ring center, and the panel tilts back about
20 degrees. The screen center is about 1.26 m above the floor.

Source: https://am.sega.jp/cad/video_game/15805 (DX cabinet, 2019-06-11).
The separate 43-inch LCD part listed in the service manual is the rectangular
display diagonal. The active touch boundary and the depth of a pressed button
relative to the glass still need a cabinet measurement.

The world origin is the screen center. Positive axes point right, up and
toward the player. Coordinates are in millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from maichart.generation.muri.model import CENTERS, DISTANCES, PADS


# The packaged MiaCode real_path_samples start and end at radius 240 on the
# virtual screen. Sensor centers use the same axes, but have smaller radii.
_NATIVE_PATH_RADIUS = 240.0


@dataclass(frozen=True)
class MachineProfile:
    button_radius_mm: float = 320.0
    screen_radius_mm: float = 260.0
    tilt_deg_from_vertical: float = 20.0
    screen_center_height_mm: float = 1260.0
    button_offset_mm: float = 0.0
    scale_label: str = "DX 机柜图纸估算"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.button_radius_mm) and math.isfinite(self.screen_radius_mm)
                and self.button_radius_mm > self.screen_radius_mm > 0):
            raise ValueError("机台按钮半径必须大于屏面半径，且两者均须为正有限值")
        if not -90 < self.tilt_deg_from_vertical < 90:
            raise ValueError("机台屏面相对竖直的倾角须在 -90° 与 90° 之间")
        if not math.isfinite(self.button_offset_mm):
            raise ValueError("按钮相对屏面的偏移须为有限值")
        if not math.isfinite(self.screen_center_height_mm) or self.screen_center_height_mm <= 0:
            raise ValueError("屏面中心离地高度须为正有限值")


_DX_PROFILE = MachineProfile()


def machine_profile() -> MachineProfile:
    return _DX_PROFILE


def _plane_xyz(x_mm: float, up_mm: float, profile: MachineProfile,
               normal_mm: float = 0.0) -> tuple[float, float, float]:
    angle = math.radians(profile.tilt_deg_from_vertical)
    sine, cosine = math.sin(angle), math.cos(angle)
    return x_mm, up_mm * cosine + normal_mm * sine, -up_mm * sine + normal_mm * cosine


def button_xyz(lane: int | str, *, profile: MachineProfile = _DX_PROFILE) -> tuple[float, float, float]:
    """Locate an outer physical button by its 1-based lane."""
    pad = f"A{lane}"
    if pad not in CENTERS:
        raise ValueError(f"未知按钮：{lane}")
    x, y = CENTERS[pad]
    scale = profile.button_radius_mm / DISTANCES["A"]
    return _plane_xyz(x * scale, -y * scale, profile, profile.button_offset_mm)


def slide_xyz(x: float, y: float, *, profile: MachineProfile = _DX_PROFILE) -> tuple[float, float, float]:
    """Convert raw MiaCode path coordinates (x right, y down) to the glass."""
    scale = profile.screen_radius_mm / _NATIVE_PATH_RADIUS
    return _plane_xyz(float(x) * scale, -float(y) * scale, profile)


def touch_xyz(pad: str, *, profile: MachineProfile = _DX_PROFILE) -> tuple[float, float, float]:
    """Locate a logical touch sensor center on the glass."""
    name = str(pad).upper()
    if name not in CENTERS:
        raise ValueError(f"未知触区：{pad}")
    return slide_xyz(*CENTERS[name], profile=profile)


def machine_payload(*, profile: MachineProfile = _DX_PROFILE) -> dict:
    """Serialize dimensions and both input surfaces for the preview."""
    buttons = [dict(id=str(lane), kind="button", xyz=list(button_xyz(lane, profile=profile)))
               for lane in range(1, 9)]
    sensors = [dict(id=pad, kind="touch", xyz=list(touch_xyz(pad, profile=profile))) for pad in PADS]
    return {
        "button_radius_mm": profile.button_radius_mm,
        "screen_radius_mm": profile.screen_radius_mm,
        "tilt_deg_from_vertical": profile.tilt_deg_from_vertical,
        "screen_center_height_mm": profile.screen_center_height_mm,
        "button_offset_mm": profile.button_offset_mm,
        "scale_label": profile.scale_label,
        "pads": buttons + sensors,
    }
