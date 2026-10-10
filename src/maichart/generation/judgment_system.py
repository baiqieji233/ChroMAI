"""舞萌DX判定区和判定机制。

舞萌的判定系统包含时间判定窗口和物理判定区域两个维度。
这对于理解真实的冲突检测至关重要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class JudgmentTiming:
    """舞萌DX判定时间窗口。
    
    不同判定等级对应的时间窗口（以Critical Perfect为基准）：
    - Critical Perfect: ±16.7ms（最严格）
    - Perfect: ±33.3ms
    - Great: ±66.7ms  
    - Good: ±100ms
    - Miss: 超出Good窗口
    
    实际判定时，游戏会在这些窗口内检测输入。
    """
    
    CRITICAL_PERFECT_MS = 16.7
    PERFECT_MS = 33.3
    GREAT_MS = 66.7
    GOOD_MS = 100.0
    
    @staticmethod
    def judgment_window_sec(grade: Literal["cp", "perfect", "great", "good"] = "perfect") -> float:
        """返回判定窗口半径（秒）。"""
        windows = {
            "cp": JudgmentTiming.CRITICAL_PERFECT_MS / 1000.0,
            "perfect": JudgmentTiming.PERFECT_MS / 1000.0,
            "great": JudgmentTiming.GREAT_MS / 1000.0,
            "good": JudgmentTiming.GOOD_MS / 1000.0,
        }
        return windows[grade]
    
    @staticmethod
    def can_judge_simultaneously(time1: float, time2: float, window_ms: float = 33.3) -> bool:
        """判断两个音符是否可能被同时判定（时间窗口重叠）。"""
        return abs(time1 - time2) * 1000 < window_ms


@dataclass(frozen=True)
class JudgmentZone:
    """舞萌DX的判定区域概念。
    
    舞萌的判定系统不仅看时间，还要看物理空间：
    
    1. **按键判定区**：
       - 每个按键（1-8）有独立的圆形判定区
       - 半径约为按键可视大小的1.2倍
       - 按键判定区之间有小间隙，不完全重叠
    
    2. **Touch判定区**：
       - A-E区域是同心环，从中心向外扩展
       - A区(中心圆)覆盖整个中心区域
       - B-E区是环形，外圈逐渐扩大
       - Touch判定区与按键判定区在空间上有重叠（尤其是A区与1号按键）
    
    3. **Slide星星判定**：
       - 星星在轨迹上移动，有一个跟随的判定区
       - 星星判定区的大小约为按键的0.8倍
       - 当星星经过按键时，如果同时有Tap输入，会产生判定冲突
    
    4. **判定优先级**：
       - Slide星星 > Tap/Hold（星星经过时，该按键的Tap会被阻挡）
       - Touch与Button可以同时判定，但A区与1号按键重叠时可能混淆
       - Hold持续期间，该按键的Tap判定被阻挡
    """
    
    # 按键判定区半径（归一化，按键中心到判定区边界的距离）
    BUTTON_ZONE_RADIUS = 0.15
    
    # 星星判定区半径
    STAR_ZONE_RADIUS = 0.12
    
    # Touch区域半径（从中心向外）
    TOUCH_A_RADIUS = 0.3   # A区：中心圆
    TOUCH_B_RADIUS = 0.5   # B区：到这个半径为止
    TOUCH_C_RADIUS = 0.7   # C区
    TOUCH_D_RADIUS = 0.9   # D区
    TOUCH_E_RADIUS = 1.1   # E区（外圈）
    
    @staticmethod
    def touch_zone_radius(zone: str) -> tuple[float, float]:
        """返回Touch区域的内外半径。
        
        返回 (inner_radius, outer_radius)
        """
        zones = {
            "A": (0.0, JudgmentZone.TOUCH_A_RADIUS),
            "B": (JudgmentZone.TOUCH_A_RADIUS, JudgmentZone.TOUCH_B_RADIUS),
            "C": (JudgmentZone.TOUCH_B_RADIUS, JudgmentZone.TOUCH_C_RADIUS),
            "D": (JudgmentZone.TOUCH_C_RADIUS, JudgmentZone.TOUCH_D_RADIUS),
            "E": (JudgmentZone.TOUCH_D_RADIUS, JudgmentZone.TOUCH_E_RADIUS),
        }
        return zones.get(zone.upper(), (0.0, 0.0))
    
    @staticmethod
    def button_zone_overlaps_touch(button: int, touch_zone: str) -> bool:
        """判断按键判定区是否与Touch区域重叠。
        
        重要：A区（中心圆）与所有按键都有可能重叠，
        但主要影响1号和8号按键（靠近上方）。
        """
        from maichart.generation.maimai_physics import ButtonLayout
        
        # 获取按键的位置（角度和半径）
        # 按键在半径1.0的圆上
        button_radius = 1.0
        button_angle_rad = ButtonLayout.button_angle(button) * 3.14159 / 180.0
        button_x = button_radius * np.cos(button_angle_rad)
        button_y = button_radius * np.sin(button_angle_rad)
        button_distance_to_center = (button_x**2 + button_y**2)**0.5
        
        # 获取Touch区域范围
        touch_inner, touch_outer = JudgmentZone.touch_zone_radius(touch_zone)
        
        # 检查按键判定区是否与Touch环重叠
        # 按键判定区：距离按键中心BUTTON_ZONE_RADIUS以内
        # 按键中心在距离中心button_distance_to_center处
        
        # 按键判定区的最近点到中心的距离
        button_zone_min = button_distance_to_center - JudgmentZone.BUTTON_ZONE_RADIUS
        # 按键判定区的最远点到中心的距离
        button_zone_max = button_distance_to_center + JudgmentZone.BUTTON_ZONE_RADIUS
        
        # 判断环形重叠：[button_zone_min, button_zone_max] 与 [touch_inner, touch_outer] 是否相交
        return not (button_zone_max < touch_inner or button_zone_min > touch_outer)
    
    @staticmethod
    def get_critical_touch_button_pairs() -> list[tuple[str, int]]:
        """返回最容易产生判定冲突的Touch-Button配对。
        
        基于实际游戏经验：
        - A区与1号按键：高度重叠，容易混淆
        - A区与8号按键：部分重叠
        - B区与多个按键：可能重叠但影响较小
        """
        return [
            ("A", 1),   # 最严重
            ("A", 8),   # 次严重
            ("B", 1),   # 可能影响
            ("B", 8),   # 可能影响
        ]


@dataclass
class JudgmentConflictRule:
    """判定冲突规则汇总。
    
    整合时间窗口和空间判定区的冲突规则。
    """
    
    @staticmethod
    def check_hold_tap_conflict(
        hold_button: int,
        hold_start: float,
        hold_duration: float,
        tap_button: int,
        tap_time: float,
        time_tolerance_ms: float = 10.0
    ) -> bool:
        """检查Hold与Tap的判定冲突。
        
        规则：
        1. 必须是同一按键
        2. Tap时间在Hold持续期间（不含起点和终点的容差范围）
        3. 原因：Hold持续期间，该按键的判定区被占用
        """
        if hold_button != tap_button:
            return False
        
        hold_end = hold_start + hold_duration
        tolerance_sec = time_tolerance_ms / 1000.0
        
        # Tap在Hold持续期间（不含边界容差）
        return hold_start + tolerance_sec < tap_time < hold_end - tolerance_sec
    
    @staticmethod
    def check_slide_tap_conflict(
        slide_star_button: int | None,
        slide_star_time: float,
        tap_button: int,
        tap_time: float,
        time_window_ms: float = 50.0,
        spatial_tolerance: float = 0.1
    ) -> bool:
        """检查Slide星星与Tap的判定冲突。
        
        规则：
        1. 星星在tap时刻必须在或接近tap按键
        2. 时间上，tap判定窗口与星星通过时间重叠
        3. 空间上，星星判定区与按键判定区重叠
        
        原因：星星占用判定优先级，会"吃掉"该位置的Tap判定
        """
        if slide_star_button is None:
            return False
        
        # 时间窗口检查
        if abs(tap_time - slide_star_time) > time_window_ms / 1000.0:
            return False
        
        # 空间检查：星星是否在tap按键上
        # slide_star_button已经是通过star_position_at计算得到的位置
        return slide_star_button == tap_button
    
    @staticmethod
    def check_touch_button_conflict(
        touch_zone: str,
        touch_start: float,
        touch_duration: float,
        button: int,
        button_time: float,
        button_duration: float = 0.0,
        spatial_check: bool = True
    ) -> bool:
        """检查Touch与Button的判定冲突。
        
        规则：
        1. 时间上有重叠（Touch存在期间与Button判定窗口重叠）
        2. 空间上有重叠（Touch区域与Button判定区重叠）
        3. 主要冲突：A区与1/8号按键
        
        原因：玩家可能误触，或者判定系统混淆输入来源
        """
        # 时间重叠检查
        touch_end = touch_start + max(touch_duration, 0.1)  # Touch最小持续时间
        button_end = button_time + button_duration
        
        # 检查时间窗口是否重叠
        if button_end < touch_start or touch_end < button_time:
            return False
        
        # 空间重叠检查
        if spatial_check:
            # 只检查关键配对
            critical_pairs = JudgmentZone.get_critical_touch_button_pairs()
            if (touch_zone, button) not in critical_pairs:
                return False
        
        return True
    
    @staticmethod
    def check_simultaneous_buttons_conflict(
        buttons: list[int],
        time_window_ms: float = 50.0,
        max_span: int = 4
    ) -> bool:
        """检查同时按键的可行性。
        
        规则：
        1. 同时按键数量不超过4（双手各2个）
        2. 按键跨度不超过180度（半圆）
        3. 同手按键之间角度不超过135度（除非交叉）
        
        这是软约束，主要用于人体工学检测。
        """
        if len(buttons) > 4:
            return True  # 超过4押，可能无理
        
        if len(buttons) <= 1:
            return False
        
        from maichart.generation.maimai_physics import ButtonLayout, HandDispatch
        
        # 检查最大跨度
        angles = [ButtonLayout.button_angle(b) for b in buttons]
        angle_span = max(angles) - min(angles)
        if angle_span > 180:
            angle_span = 360 - angle_span
        
        if angle_span > 180:
            return True  # 跨度过大
        
        # 检查同手按键
        left_btns = [b for b in buttons if HandDispatch.primary_hand(b) == "left"]
        right_btns = [b for b in buttons if HandDispatch.primary_hand(b) == "right"]
        
        for hand_btns in [left_btns, right_btns]:
            if len(hand_btns) >= 2:
                hand_angles = [ButtonLayout.button_angle(b) for b in hand_btns]
                hand_span = max(hand_angles) - min(hand_angles)
                if hand_span > 135:  # 同手按键跨度限制
                    return True
        
        return False


# 用于导入
import numpy as np


if __name__ == "__main__":
    print("=== 舞萌DX判定系统测试 ===")
    
    print("\n1. 判定时间窗口：")
    for grade in ["cp", "perfect", "great", "good"]:
        window = JudgmentTiming.judgment_window_sec(grade) * 1000
        print(f"   {grade.upper():8s}: ±{window:.1f}ms")
    
    print("\n2. Touch区域半径：")
    for zone in ["A", "B", "C", "D", "E"]:
        inner, outer = JudgmentZone.touch_zone_radius(zone)
        print(f"   {zone}区: {inner:.2f} - {outer:.2f}")
    
    print("\n3. 关键Touch-Button冲突配对：")
    pairs = JudgmentZone.get_critical_touch_button_pairs()
    for touch, btn in pairs:
        print(f"   Touch {touch} + Button {btn}")
    
    print("\n4. 判定冲突示例：")
    
    # Hold-Tap冲突
    conflict = JudgmentConflictRule.check_hold_tap_conflict(
        hold_button=1, hold_start=1.0, hold_duration=2.0,
        tap_button=1, tap_time=1.5
    )
    print(f"   Hold 1 [1.0-3.0s] vs Tap 1 [1.5s]: {'冲突' if conflict else '正常'}")
    
    # Slide-Tap冲突
    conflict = JudgmentConflictRule.check_slide_tap_conflict(
        slide_star_button=3, slide_star_time=2.0,
        tap_button=3, tap_time=2.02
    )
    print(f"   Slide星星@Button3 [2.0s] vs Tap 3 [2.02s]: {'冲突' if conflict else '正常'}")
    
    # Touch-Button冲突
    conflict = JudgmentConflictRule.check_touch_button_conflict(
        touch_zone="A", touch_start=1.0, touch_duration=0.5,
        button=1, button_time=1.2
    )
    print(f"   Touch A [1.0-1.5s] vs Button 1 [1.2s]: {'冲突' if conflict else '正常'}")
    
    print("\n测试完成")
