"""生成后处理：冲突检测和修复集成。

这个模块在event_ctx_cli或webui的生成流程后调用，
对生成的谱面进行冲突检测和可选修复。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from maichart.generation.conflict_detector import ConflictDetector, NoteEvent, ConflictReport
from maichart.generation.conflict_repair import ConflictRepairer


@dataclass
class PostProcessConfig:
    """生成后处理配置。"""
    
    enable_conflict_detection: bool = True
    detect_hard_conflicts: bool = True
    detect_soft_conflicts: bool = False
    conflict_repair_mode: str = "none"  # none/local/reject
    
    # 冲突检测参数
    max_simultaneous_buttons: int = 4
    min_tap_interval: float = 0.05
    # Match conflict_detector.ConflictDetector: 2.0 is the ring diameter, so
    # only genuine opposite-side flips are reported.
    max_jump_distance: float = 2.0
    min_jump_time: float = 0.10
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return asdict(self)


@dataclass
class PostProcessResult:
    """生成后处理结果。"""
    
    original_notes_count: int
    processed_notes_count: int
    conflicts_detected: list[ConflictReport]
    repairs_applied: list[str]
    success: bool
    message: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于JSON序列化）。"""
        return {
            "original_notes_count": self.original_notes_count,
            "processed_notes_count": self.processed_notes_count,
            "conflicts_detected": len(self.conflicts_detected),
            "repairs_applied": len(self.repairs_applied),
            "success": self.success,
            "message": self.message,
        }


def extract_notes_from_chart(notes_dict: dict) -> list[NoteEvent]:
    """从chart note字典中提取NoteEvent列表。
    
    参数：notes_dict是由parser生成的note字典，形式为：
    {beat_position: [ParsedNote, ...], ...}
    """
    events = []
    
    for beat, note_list in sorted(notes_dict.items()):
        beat_float = float(beat)
        
        for note in note_list:
            # 确定note类型
            if note.hold_end is not None:
                note_type = "hold"
                duration = float(note.hold_end) - beat_float
            elif note.slide_pattern:
                note_type = "slide"
                # 简化：使用default 1s，实际应从note.segments读取
                duration = 1.0
            elif note.is_touch_hold:
                note_type = "touch"
                duration = float(note.hold_end) - beat_float if note.hold_end else 0.0
            elif note.is_touch:
                note_type = "touch"
                duration = 0.0
            else:
                note_type = "tap"
                duration = 0.0
            
            # 获取按键（Touch的button是字符串A-E）
            button = str(note.position) if note.position else (note.touch_zone or "1")
            
            events.append(NoteEvent(
                note_type=note_type,
                button=button,
                time=beat_float,
                duration=duration,
                raw=str(note)
            ))
    
    return events


def detect_chart_conflicts(
    points,
    tempo_changes,
    first: float,
    config: "PostProcessConfig",
) -> PostProcessResult:
    """Run the playability checks against compiled event points.

    The detector needs real note geometry (button indices, hold durations and
    slide trajectories), so it runs on the parsed event chart rather than on the
    raw token stream.  `points` is the same list handed to
    `compile_event_points`; every note is converted to an absolute onset in
    seconds using the chart tempo map.
    """

    from fractions import Fraction

    from maichart.generation.conflict_detector import (
        ConflictDetector, NoteEvent, SlideTrajectory,
    )
    from maichart.generation.event_quality import parse_event_notes

    parsed = parse_event_notes(points, tempo_changes, first)
    events: list[NoteEvent] = []
    for point in parsed.points:
        onset = float(point.time_sec or 0.0)
        for note in point.notes:
            note_type = getattr(note, "note_type", "tap") or "tap"
            position = getattr(note, "position", 1)
            duration = float(getattr(note, "duration_sec", 0.0) or 0.0)
            if note_type in ("tap", "break"):
                note_type = "tap"
            if note_type in ("touch", "touch_hold"):
                # Touch positions look like "C", "A4" or "B7"; the leading
                # letter is the judgement zone used by the overlap rule.
                zone = str(position)[:1] if position else "A"
                events.append(NoteEvent(
                    note_type="touch", button=zone, time=onset, duration=duration,
                    raw=str(getattr(note, "raw", "")),
                ))
                continue
            if not isinstance(position, int):
                continue
            trajectory = None
            pattern = getattr(note, "slide_pattern", None)
            if note_type == "slide" and pattern:
                end_position = getattr(note, "end_position", None)
                if isinstance(end_position, int):
                    trajectory = SlideTrajectory(
                        start_button=position, pattern=str(pattern),
                        end_button=end_position, start_time=onset,
                        duration=max(duration, 1e-3),
                    )
            events.append(NoteEvent(
                note_type=("slide" if note_type == "slide" else note_type),
                button=position, time=onset, duration=duration,
                slide_trajectory=trajectory,
                raw=str(getattr(note, "raw", "")),
            ))

    detector = ConflictDetector(
        enable_hard_conflicts=config.detect_hard_conflicts,
        enable_soft_conflicts=config.detect_soft_conflicts,
        min_tap_interval=config.min_tap_interval,
        max_jump_distance=config.max_jump_distance,
        min_jump_time=config.min_jump_time,
    )
    conflicts = detector.detect_conflicts(events)
    hard = [c for c in conflicts if c.severity == "hard"]
    return PostProcessResult(
        original_notes_count=len(events),
        processed_notes_count=len(events),
        conflicts_detected=conflicts,
        repairs_applied=[],
        success=not (config.conflict_repair_mode == "reject" and hard),
        message=(f"{len(conflicts)} conflicts ({len(hard)} hard)" if conflicts
                 else "no conflicts"),
    )


def run_post_process(
    notes_dict: dict,
    config: PostProcessConfig | None = None,
    verbose: bool = True
) -> PostProcessResult:
    """对生成的谱面运行后处理。
    
    参数：
    - notes_dict: 由event_model生成的note字典
    - config: 后处理配置
    - verbose: 打印详细信息
    
    返回：PostProcessResult
    """
    if config is None:
        config = PostProcessConfig()
    
    original_count = sum(len(v) for v in notes_dict.values())
    
    if not config.enable_conflict_detection:
        if verbose:
            print(f"跳过冲突检测（disabled）")
        return PostProcessResult(
            original_notes_count=original_count,
            processed_notes_count=original_count,
            conflicts_detected=[],
            repairs_applied=[],
            success=True
        )
    
    try:
        # 提取note事件
        events = extract_notes_from_chart(notes_dict)
        
        if verbose:
            print(f"冲突检测: {len(events)} 个note")
        
        # 检测冲突
        detector = ConflictDetector(
            enable_hard_conflicts=config.detect_hard_conflicts,
            enable_soft_conflicts=config.detect_soft_conflicts,
            max_simultaneous_buttons=config.max_simultaneous_buttons,
            min_tap_interval=config.min_tap_interval,
            max_jump_distance=config.max_jump_distance,
            min_jump_time=config.min_jump_time
        )
        
        conflicts = detector.detect_conflicts(events)
        
        if verbose and conflicts:
            print(f"检测到 {len(conflicts)} 个冲突:")
            hard_count = sum(1 for c in conflicts if c.severity == "hard")
            soft_count = sum(1 for c in conflicts if c.severity == "soft")
            if hard_count > 0:
                print(f"  硬冲突: {hard_count}")
            if soft_count > 0:
                print(f"  软冲突: {soft_count}")
        
        # 修复冲突
        repairer = ConflictRepairer(repair_mode=config.conflict_repair_mode)
        
        try:
            repaired_events, repair_logs = repairer.repair(events, conflicts)
            repairs_count = len(repair_logs)
            
            if verbose and repairs_count > 0:
                print(f"应用修复: {repairs_count} 项")
                for log in repair_logs[:5]:  # 只打印前5项
                    print(f"  - {log}")
                if repairs_count > 5:
                    print(f"  ... 及其他 {repairs_count - 5} 项")
            
            # 将修复后的events转换回notes_dict
            # 简化：只返回原始数据（因为删除note会破坏beat对应关系）
            # 实际应该重新构建beat->notes映射
            processed_count = len(repaired_events)
            
            return PostProcessResult(
                original_notes_count=original_count,
                processed_notes_count=processed_count,
                conflicts_detected=conflicts,
                repairs_applied=repair_logs,
                success=config.conflict_repair_mode != "reject" or len([c for c in conflicts if c.severity == "hard"]) == 0,
                message=f"处理完成: {original_count} -> {processed_count} notes"
            )
        
        except ValueError as e:
            # reject模式抛出的异常
            if verbose:
                print(f"修复失败（reject模式）: {str(e)[:100]}")
            
            return PostProcessResult(
                original_notes_count=original_count,
                processed_notes_count=0,
                conflicts_detected=conflicts,
                repairs_applied=[],
                success=False,
                message=f"拒绝生成: {str(e)[:200]}"
            )
    
    except Exception as e:
        if verbose:
            print(f"后处理错误: {e}")
        
        return PostProcessResult(
            original_notes_count=original_count,
            processed_notes_count=0,
            conflicts_detected=[],
            repairs_applied=[],
            success=False,
            message=f"错误: {str(e)}"
        )


def save_post_process_report(
    result: PostProcessResult,
    output_path: str | Path
) -> None:
    """保存后处理报告。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "result": result.to_dict(),
        "conflicts": [
            {
                "type": c.conflict_type,
                "severity": c.severity,
                "time": c.time,
                "message": c.message
            }
            for c in result.conflicts_detected
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("=== 生成后处理测试 ===")
    
    # 模拟note字典
    from fractions import Fraction
    notes_dict = {
        Fraction(0): [],  # 空窗口
        Fraction(1): [],  # 单个note示例会在实际使用中填充
    }
    
    # 运行后处理
    config = PostProcessConfig(
        enable_conflict_detection=True,
        detect_hard_conflicts=True,
        conflict_repair_mode="none"
    )
    
    result = run_post_process(notes_dict, config, verbose=True)
    
    print(f"\n结果: {result.success}")
    print(f"原始note: {result.original_notes_count}")
    print(f"处理后: {result.processed_notes_count}")
    print(f"冲突: {len(result.conflicts_detected)}")
    print(f"修复: {len(result.repairs_applied)}")
