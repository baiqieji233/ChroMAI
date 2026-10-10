"""舞萌DX物理约束模型。

根据真实maimai游戏机制构建的物理模型，用于可玩性检测。
不过度模拟人体工学，只检测游戏判定系统的真实冲突。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ButtonLayout:
    """舞萌DX 1-8号按键在八边形上的物理布局。
    
    按键布局（修正版）：
    - 1号按键位于正上方顺时针偏转22.5度（八边形第一个顶点）
    - 按键按顺时针方向每隔45度排列
    - 八边形上边由1号和8号按键构成，与地面平行
    
    标准角度坐标系：0度为正右方，逆时针为正方向
    """
    
    @staticmethod
    def button_angle(button: int) -> float:
        """返回按键的角度（度数，0度为正右方，逆时针为正）。
        
        舞萌布局：1号在正上方偏右22.5°，顺时针递减45°。
        转换为标准坐标系：
        - 1号：90° - 22.5° = 67.5°
        - 2号：67.5° - 45° = 22.5°
        - 3号：-22.5° (或 337.5°)
        - 4号：-67.5° (或 292.5°)
        - 5号：-112.5° (或 247.5°)
        - 6号：-157.5° (或 202.5°)
        - 7号：157.5°
        - 8号：112.5°
        """
        if not (1 <= button <= 8):
            raise ValueError(f"按键必须在1-8范围内，收到 {button}")
        angle = 67.5 - (button - 1) * 45.0
        return angle if angle >= 0 else angle + 360
    
    @staticmethod
    def angular_distance(btn1: int, btn2: int) -> float:
        """计算两按键的最短角度距离（度）。"""
        angle1 = ButtonLayout.button_angle(btn1)
        angle2 = ButtonLayout.button_angle(btn2)
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
    
    @staticmethod
    def physical_distance(btn1: int, btn2: int, radius: float = 1.0) -> float:
        """计算两按键的物理直线距离（归一化半径=1）。
        
        使用弦长公式：d = 2r * sin(θ/2)
        """
        ang_dist = math.radians(ButtonLayout.angular_distance(btn1, btn2))
        return 2 * radius * math.sin(ang_dist / 2)
    
    @staticmethod
    def is_adjacent(btn1: int, btn2: int) -> bool:
        """判断两按键是否相邻（45度间隔）。"""
        return ButtonLayout.angular_distance(btn1, btn2) == 45.0


@dataclass(frozen=True)
class HandDispatch:
    """双手按键分配模型。
    
    基础规则（根据真实maimai布局）：
    - 左手优先：4, 5, 6, 7（底部半圆+左侧）
    - 右手优先：1, 2, 3, 8（顶部半圆+右侧）
    
    特殊情况：
    - Slide轨迹期间，起始手保持追踪星星
    - Touch可用任意手指，不占用tap手指
    - 大跨度跳跃时可能需要交叉手
    """
    
    LEFT_PRIMARY = frozenset({4, 5, 6, 7})
    RIGHT_PRIMARY = frozenset({1, 2, 3, 8})
    
    @staticmethod
    def primary_hand(button: int) -> Literal["left", "right"]:
        """返回该按键的主要负责手。"""
        return "left" if button in HandDispatch.LEFT_PRIMARY else "right"
    
    @staticmethod
    def can_reach_simultaneous(buttons: list[int], allow_cross: bool = False) -> bool:
        """判断多个按键能否同时按下。
        
        基础规则：
        - 左手按键+右手按键 → 可以
        - 同手2个相邻或接近按键 → 可以（伸展手指）
        - 同手2个按键且角度≥90° → 需要allow_cross
        - 同手3+按键 → 需要allow_cross
        """
        if len(buttons) <= 1:
            return True
        
        left_btns = [b for b in buttons if b in HandDispatch.LEFT_PRIMARY]
        right_btns = [b for b in buttons if b in HandDispatch.RIGHT_PRIMARY]
        
        # 双手各持1-2个且不过度分散 → 合理
        if len(left_btns) <= 2 and len(right_btns) <= 2:
            # 检查同手按键是否过度分散
            for hand_btns in [left_btns, right_btns]:
                if len(hand_btns) == 2:
                    span = ButtonLayout.angular_distance(hand_btns[0], hand_btns[1])
                    if span > 90 and not allow_cross:
                        return False
            return True
        
        # 单手3+ → 需要交叉或特殊技巧
        if len(left_btns) >= 3 or len(right_btns) >= 3:
            return allow_cross
        
        return True
    
    @staticmethod
    def estimate_transition_time(from_btn: int, to_btn: int) -> float:
        """估算从一个按键移动到另一个按键的最小时间（秒）。
        
        基于物理距离的粗略估算：
        - 相邻按键：~0.08s
        - 对角按键：~0.15s
        - 跨手切换：额外+0.03s
        """
        base_time = ButtonLayout.physical_distance(from_btn, to_btn) * 0.08
        
        # 跨手切换惩罚
        if HandDispatch.primary_hand(from_btn) != HandDispatch.primary_hand(to_btn):
            base_time += 0.03
        
        return base_time


if __name__ == "__main__":
    # 简单验证
    print("=== 按键布局验证 ===")
    for i in range(1, 9):
        print(f"按键 {i}: 角度 {ButtonLayout.button_angle(i):.1f}°, 主手: {HandDispatch.primary_hand(i)}")
    
    print(f"\n按键1到5的角度距离: {ButtonLayout.angular_distance(1, 5):.1f}°")
    print(f"按键1到5的物理距离: {ButtonLayout.physical_distance(1, 5):.3f}")
    print(f"按键1到2是否相邻: {ButtonLayout.is_adjacent(1, 2)}")
    
    print(f"\n同时按[1,2,3]可行性（不交叉）: {HandDispatch.can_reach_simultaneous([1, 2, 3], False)}")
    print(f"同时按[1,5]可行性: {HandDispatch.can_reach_simultaneous([1, 5], False)}")
