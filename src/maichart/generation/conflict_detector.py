"""舞萌谱面冲突检测器。

检测舞萌DX谱面中的无理配置，基于真实游戏判定机制。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from maichart.generation.maimai_physics import ButtonLayout, HandDispatch


@dataclass
class SlideTrajectory:
    """Slide星星轨迹模型。
    
    星星在slide持续期间从起始按键连续移动到目标按键。
    """
    
    start_button: int
    pattern: str
    end_button: int
    start_time: float  # 秒
    duration: float    # 秒
    
    def angular_span(self) -> float:
        """Signed sweep in degrees for the pattern, or 0 when unknown.

        ``-`` is the direct route and may be the long way round for a
        half-turn; ``<``/``>`` are the explicit counter-clockwise / clockwise
        arcs, so their sign is fixed.  A 180-degree ``-`` is genuinely
        ambiguous (both routes are equally long), and returning 0 lets callers
        tell "no sweep" from "half turn".
        """

        start = ButtonLayout.button_angle(self.start_button)
        end = ButtonLayout.button_angle(self.end_button)
        ccw = (end - start) % 360.0
        cw = ccw - 360.0
        if self.pattern == "<":
            return ccw
        if self.pattern == ">":
            return cw
        if abs(abs(ccw) - 180.0) < 1e-6:
            return 0.0
        return ccw if abs(ccw) <= abs(cw) else cw

    def star_position_at(self, time: float) -> int | None:
        """返回time时刻星星所在的按键位置。
        
        简化模型：
        - 直线slide (-)：按角度线性插值
        - 弧线slide (<, >, ^)：按弧长线性插值
        - 复杂轨迹：分段近似
        
        返回None表示星星不在任何按键上（轨迹中间）。
        只有在星星非常接近按键时才返回按键编号。
        """
        if time < self.start_time or time > self.start_time + self.duration:
            return None
        
        # 起点和终点时刻
        if abs(time - self.start_time) < 0.01:
            return self.start_button
        if abs(time - (self.start_time + self.duration)) < 0.01:
            return self.end_button
        
        progress = (time - self.start_time) / self.duration
        
        # 直线或弧线slide：线性插值角度
        if self.pattern in ("-", "<", ">", "^"):
            start_ang = ButtonLayout.button_angle(self.start_button)
            span = self.angular_span()
            if span == 0.0 and self.start_button != self.end_button:
                # A half-turn without a direction: the star crosses the middle
                # of the board whichever way it goes, so no button can be said
                # to be "occupied" with confidence.  Reporting None is the
                # conservative choice and avoids flagging the three buttons the
                # straight route happens to pass; a real conflict there would
                # need the authored path geometry, not a straight-line guess.
                return None
            current_ang = (start_ang + progress * span) % 360

            # Only treat the star as sitting on a button when it is close to it.
            # A star's judgement disc is much smaller than the 45-degree button
            # pitch, so a wide tolerance invents collisions at the midpoint of
            # every slide.  Six degrees is about a seventh of the pitch.
            min_dist = float('inf')
            closest_btn = None
            for btn in range(1, 9):
                btn_ang = ButtonLayout.button_angle(btn)
                ang_diff = min(abs(current_ang - btn_ang),
                               360 - abs(current_ang - btn_ang))
                if ang_diff < min_dist:
                    min_dist = ang_diff
                    closest_btn = btn
            if min_dist < 6.0:
                return closest_btn
            return None
        
        # 复杂轨迹：简化处理，只考虑起点和终点
        if progress < 0.2:
            return self.start_button
        elif progress > 0.8:
            return self.end_button
        else:
            return None


@dataclass
class NoteEvent:
    """单个note事件。"""
    note_type: Literal["tap", "hold", "slide", "touch"]
    button: int | str  # 按键1-8 或 touch区域A-E
    time: float  # 起音时间（秒）
    duration: float = 0.0  # hold/slide持续时间
    slide_trajectory: SlideTrajectory | None = None
    raw: str = ""  # 原始note文本（用于报告）


@dataclass
class ConflictReport:
    """冲突检测报告。"""
    has_conflict: bool
    conflict_type: str
    time: float
    notes: list[NoteEvent]
    severity: Literal["hard", "soft"]
    message: str


class ConflictDetector:
    """检测舞萌谱面中的无理配置。
    
    检测内容：
    - 硬冲突：游戏判定系统无法处理的配置
    - 软冲突：人体工学上困难但非绝对不可能的配置
    
    设计原则（根据用户要求）：
    - 所有检测均为可选，可通过参数启用/禁用
    - 不过度检测：Touch多押是合理的
    - 基于真实游戏判定机制，而非过度拟合人体工学
    """
    
    def __init__(
        self,
        enable_hard_conflicts: bool = True,
        enable_soft_conflicts: bool = False,
        max_simultaneous_buttons: int = 4,
        min_tap_interval: float = 0.05,  # 50ms最小间隔
        # A jump is only "hard on the hands" when it is both long and fast.
        # 2.0 is the full diameter of the ring (the 1-5 / 3-7 diagonal), so the
        # reach rule fires on genuine opposite-side flips rather than on the
        # ordinary 135-degree movement of 16th notes at ~180 BPM, which a
        # competent player reads without trouble.
        max_jump_distance: float = 2.0,  # 最大跳跃距离
        min_jump_time: float = 0.10,     # 大跳跃最小时间
    ):
        """
        参数：
        - enable_hard_conflicts: 启用硬冲突检测（推荐启用）
        - enable_soft_conflicts: 启用软冲突检测（人体工学）
        - max_simultaneous_buttons: 最大同时按键数
        - min_tap_interval: 最小tap间隔（秒）
        - max_jump_distance: 触发软冲突的跳跃距离
        - min_jump_time: 大跳跃的最小允许时间
        """
        self.enable_hard = enable_hard_conflicts
        self.enable_soft = enable_soft_conflicts
        self.max_simultaneous = max_simultaneous_buttons
        self.min_interval = min_tap_interval
        self.max_jump_distance = max_jump_distance
        self.min_jump_time = min_jump_time
    
    def detect_conflicts(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测notes中的所有冲突。"""
        conflicts = []
        
        if self.enable_hard:
            conflicts.extend(self._detect_hold_tap_overlap(notes))
            conflicts.extend(self._detect_slide_tap_collision(notes))
            conflicts.extend(self._detect_touch_button_overlap(notes))
        
        if self.enable_soft:
            conflicts.extend(self._detect_excessive_jumps(notes))
            conflicts.extend(self._detect_hand_dispatch_conflict(notes))
            conflicts.extend(self._detect_rapid_alternation(notes))
        
        return conflicts
    
    def _detect_hold_tap_overlap(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测Hold与Tap在同一按键的时间重叠。
        
        规则：Hold A [t1, t2] 期间，Tap A 出现在 t ∈ (t1, t2) → 硬冲突
        """
        conflicts = []
        
        holds = [n for n in notes if n.note_type == "hold"]
        taps = [n for n in notes if n.note_type == "tap"]
        
        for hold in holds:
            if not isinstance(hold.button, int):
                continue
            hold_end = hold.time + hold.duration
            
            for tap in taps:
                if not isinstance(tap.button, int):
                    continue
                
                if hold.button == tap.button:
                    # 检查tap是否在hold持续期间（不包括边界）
                    if hold.time < tap.time < hold_end - 0.01:  # 10ms容差
                        conflicts.append(ConflictReport(
                            has_conflict=True,
                            conflict_type="hold_tap_overlap",
                            time=tap.time,
                            notes=[hold, tap],
                            severity="hard",
                            message=f"Hold {hold.button} [{hold.time:.2f}s-{hold_end:.2f}s] 与 Tap {tap.button} [{tap.time:.2f}s] 判定冲突"
                        ))
        
        return conflicts
    
    def _detect_slide_tap_collision(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测Slide星星与Tap的位置冲突。
        
        规则：Tap时刻星星恰好在该按键上 → 硬冲突
        """
        conflicts = []
        
        slides = [n for n in notes if n.note_type == "slide" and n.slide_trajectory]
        taps = [n for n in notes if n.note_type == "tap" and isinstance(n.button, int)]
        
        for slide in slides:
            traj = slide.slide_trajectory
            if not traj:
                continue
            
            for tap in taps:
                # 计算tap时刻星星的位置
                star_btn = traj.star_position_at(tap.time)
                
                # 只有星星恰好在tap按键上时才冲突
                if star_btn == tap.button:
                    conflicts.append(ConflictReport(
                        has_conflict=True,
                        conflict_type="slide_tap_collision",
                        time=tap.time,
                        notes=[slide, tap],
                        severity="hard",
                        message=f"Slide {traj.start_button}{traj.pattern}{traj.end_button} 星星在 {tap.time:.2f}s 位于按键 {star_btn}，与 Tap {tap.button} 冲突"
                    ))
        
        return conflicts
    
    def _detect_touch_button_overlap(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测Touch与Button的判定冲突。
        
        根据用户注释：
        - Touch多押是合理的，不检测
        - 只检测：Touch A区与1号按键的时间重叠（判定混淆）
        """
        conflicts = []
        
        touches = [n for n in notes if n.note_type == "touch"]
        buttons = [n for n in notes if n.note_type in ("tap", "hold") and isinstance(n.button, int)]
        
        # 只检查Touch A区与1号按键的严重冲突
        for touch in touches:
            if touch.button != "A":
                continue
            
            touch_end = touch.time + touch.duration if touch.duration > 0 else touch.time + 0.1
            
            for btn_note in buttons:
                if btn_note.button == 1:  # A区主要影响1号按键
                    # 时间窗口重叠检测
                    btn_end = btn_note.time + btn_note.duration if btn_note.duration > 0 else btn_note.time
                    
                    # 检查时间重叠
                    if not (btn_end < touch.time or touch_end < btn_note.time):
                        conflicts.append(ConflictReport(
                            has_conflict=True,
                            conflict_type="touch_button_overlap",
                            time=btn_note.time,
                            notes=[touch, btn_note],
                            severity="hard",
                            message=f"Touch A区 [{touch.time:.2f}s] 与 按键1的 {btn_note.note_type} [{btn_note.time:.2f}s] 判定可能冲突"
                        ))
        
        return conflicts
    
    def _detect_excessive_jumps(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测过大的按键跳跃（软冲突）。
        
        规则：短时间内大跨度跳跃可能超出人类反应速度
        """
        conflicts = []
        
        taps = sorted(
            [n for n in notes if n.note_type == "tap" and isinstance(n.button, int)], 
            key=lambda n: n.time
        )
        
        for i in range(len(taps) - 1):
            t1, t2 = taps[i], taps[i + 1]
            time_diff = t2.time - t1.time
            # A zero or near-zero gap is a chord, not a jump: two buttons
            # pressed at the same instant are one gesture per hand, and the
            # distance between them is a reach, which ``HandDispatch`` covers.
            # Without this the detector flags every legitimate chord.
            if time_diff < 1e-3:
                continue
            distance = ButtonLayout.physical_distance(t1.button, t2.button)
            
            # 短时间内大跨度跳跃
            if time_diff < self.min_jump_time and distance > self.max_jump_distance:
                conflicts.append(ConflictReport(
                    has_conflict=True,
                    conflict_type="excessive_jump",
                    time=t2.time,
                    notes=[t1, t2],
                    severity="soft",
                    message=f"按键 {t1.button}→{t2.button} 在 {time_diff*1000:.0f}ms 内跳跃，物理距离 {distance:.2f} 过大"
                ))
        
        return conflicts
    
    def _detect_hand_dispatch_conflict(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测双手调度冲突（软冲突）。
        
        规则：同一时刻的多押可能超出单手范围
        """
        conflicts = []
        
        # 按时间分组
        time_groups: dict[float, list[NoteEvent]] = {}
        for note in notes:
            if note.note_type in ("tap", "hold") and isinstance(note.button, int):
                t_key = round(note.time, 3)  # 3ms容差
                if t_key not in time_groups:
                    time_groups[t_key] = []
                time_groups[t_key].append(note)
        
        for t, group in time_groups.items():
            if len(group) > 1:
                buttons = [n.button for n in group if isinstance(n.button, int)]
                if len(buttons) > 1 and not HandDispatch.can_reach_simultaneous(buttons, allow_cross=False):
                    conflicts.append(ConflictReport(
                        has_conflict=True,
                        conflict_type="hand_dispatch_conflict",
                        time=t,
                        notes=group,
                        severity="soft",
                        message=f"{len(buttons)}押配置 {buttons} 在 {t:.2f}s 可能超出双手范围"
                    ))
        
        return conflicts
    
    def _detect_rapid_alternation(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """Detect sustained hand alternation that is faster than the corpus.

        A left-right-left trill is ordinary charting: at 180 BPM an eighth-note
        trill alternates every 167 ms and a sixteenth-note one every 83 ms, and
        both are played comfortably.  What is actually worth flagging is a
        *sustained* run -- at least five alternating taps whose average gap is
        under 100 ms (roughly 16th notes above 180 BPM) -- because that is where
        players start losing the pattern rather than reading it.
        """

        conflicts = []
        taps = sorted(
            [n for n in notes if n.note_type == "tap" and isinstance(n.button, int)],
            key=lambda n: n.time,
        )
        run: list[NoteEvent] = []
        for note in taps:
            if run and HandDispatch.primary_hand(note.button) == HandDispatch.primary_hand(run[-1].button):
                run = []
            run.append(note)
            if len(run) < 5:
                continue
            span = run[-1].time - run[0].time
            if span > 0 and (span / (len(run) - 1)) < 0.10:
                conflicts.append(ConflictReport(
                    has_conflict=True,
                    conflict_type="rapid_alternation",
                    time=run[0].time,
                    notes=list(run),
                    severity="soft",
                    message=(f"连续{len(run)}个左右手交替在{span*1000:.0f}ms内完成"
                             f"（平均{span/(len(run)-1)*1000:.0f}ms/个），可能难以稳定读取"),
                ))
                run = []
        return conflicts


if __name__ == "__main__":
    # 简单测试
    print("=== 冲突检测器测试 ===")
    
    detector = ConflictDetector(enable_hard_conflicts=True, enable_soft_conflicts=True)
    
    # 测试1: Hold-Tap冲突
    notes = [
        NoteEvent("hold", 1, 1.0, 2.0),
        NoteEvent("tap", 1, 1.5),
    ]
    conflicts = detector.detect_conflicts(notes)
    print(f"\n测试1 (Hold-Tap冲突): {len(conflicts)} 个冲突")
    for c in conflicts:
        print(f"  - {c.conflict_type}: {c.message}")
    
    # 测试2: 正常配置
    notes = [
        NoteEvent("tap", 1, 1.0),
        NoteEvent("tap", 2, 1.2),
        NoteEvent("tap", 3, 1.4),
    ]
    conflicts = detector.detect_conflicts(notes)
    print(f"\n测试2 (正常序列): {len(conflicts)} 个冲突")
    
    print("\n测试完成")
