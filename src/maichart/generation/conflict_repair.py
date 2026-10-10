"""冲突修复器（可选功能）。

根据用户要求：
- 修复是可选的，可以选择"只检测不修复"
- 修复可以是局部的（单个窗口）或全局拒绝
"""

from __future__ import annotations

from typing import Literal

from maichart.generation.conflict_detector import ConflictReport, NoteEvent


class ConflictRepairer:
    """修复检测到的冲突。"""
    
    def __init__(self, repair_mode: Literal["none", "local", "reject"] = "none"):
        """
        参数：
        - none: 只检测，不修复
        - local: 局部修复冲突窗口（删除或移动冲突note）
        - reject: 检测到硬冲突时拒绝整个生成
        """
        self.mode = repair_mode
    
    def repair(
        self, 
        notes: list[NoteEvent], 
        conflicts: list[ConflictReport]
    ) -> tuple[list[NoteEvent], list[str]]:
        """根据mode修复冲突。
        
        返回：
        - 修复后的notes列表
        - 修复日志列表
        """
        if self.mode == "none":
            return notes, []
        
        if self.mode == "reject":
            hard_conflicts = [c for c in conflicts if c.severity == "hard"]
            if hard_conflicts:
                messages = [f"{c.conflict_type}: {c.message}" for c in hard_conflicts]
                raise ValueError(
                    f"检测到 {len(hard_conflicts)} 个硬冲突，拒绝生成：\n" + 
                    "\n".join(messages)
                )
            return notes, []
        
        if self.mode == "local":
            return self._repair_local(notes, conflicts)
        
        return notes, []
    
    def _repair_local(
        self, 
        notes: list[NoteEvent], 
        conflicts: list[ConflictReport]
    ) -> tuple[list[NoteEvent], list[str]]:
        """局部修复：删除或移动冲突note。
        
        修复策略：
        - Hold-Tap重叠：删除tap
        - Slide-Tap碰撞：删除tap
        - Touch-Button重叠：删除touch（保留button）
        - 软冲突：只记录，不删除
        """
        to_remove = set()
        repair_log = []
        
        for conflict in conflicts:
            if conflict.severity == "hard":
                # 修复硬冲突
                if conflict.conflict_type == "hold_tap_overlap":
                    # 删除tap，保留hold
                    for note in conflict.notes:
                        if note.note_type == "tap":
                            to_remove.add(id(note))
                            repair_log.append(
                                f"删除 Tap {note.button} @ {note.time:.2f}s (与Hold重叠)"
                            )
                            break
                
                elif conflict.conflict_type == "slide_tap_collision":
                    # 删除tap，保留slide
                    for note in conflict.notes:
                        if note.note_type == "tap":
                            to_remove.add(id(note))
                            repair_log.append(
                                f"删除 Tap {note.button} @ {note.time:.2f}s (与Slide星星碰撞)"
                            )
                            break
                
                elif conflict.conflict_type == "touch_button_overlap":
                    # 删除touch，保留button
                    for note in conflict.notes:
                        if note.note_type == "touch":
                            to_remove.add(id(note))
                            repair_log.append(
                                f"删除 Touch {note.button} @ {note.time:.2f}s (与Button重叠)"
                            )
                            break
            
            elif conflict.severity == "soft":
                # 软冲突只记录，不修复
                repair_log.append(
                    f"软冲突警告: {conflict.conflict_type} @ {conflict.time:.2f}s"
                )
        
        repaired_notes = [n for n in notes if id(n) not in to_remove]
        return repaired_notes, repair_log


if __name__ == "__main__":
    # 简单测试
    print("=== 冲突修复器测试 ===")
    
    from maichart.generation.conflict_detector import ConflictDetector
    
    # 测试场景：Hold-Tap冲突
    notes = [
        NoteEvent("hold", 1, 1.0, 2.0),
        NoteEvent("tap", 1, 1.5),
        NoteEvent("tap", 2, 1.6),
    ]
    
    detector = ConflictDetector(enable_hard_conflicts=True)
    conflicts = detector.detect_conflicts(notes)
    
    print(f"\n原始notes: {len(notes)} 个")
    print(f"检测到冲突: {len(conflicts)} 个")
    
    # 模式1: 只检测
    repairer_none = ConflictRepairer(repair_mode="none")
    repaired, log = repairer_none.repair(notes, conflicts)
    print(f"\n模式'none': {len(repaired)} 个notes保留")
    
    # 模式2: 局部修复
    repairer_local = ConflictRepairer(repair_mode="local")
    repaired, log = repairer_local.repair(notes, conflicts)
    print(f"\n模式'local': {len(repaired)} 个notes保留")
    for msg in log:
        print(f"  - {msg}")
    
    # 模式3: 拒绝
    repairer_reject = ConflictRepairer(repair_mode="reject")
    try:
        repaired, log = repairer_reject.repair(notes, conflicts)
        print(f"\n模式'reject': 未触发拒绝")
    except ValueError as e:
        print(f"\n模式'reject': 拒绝生成")
        print(f"  原因: {str(e)[:100]}...")
    
    print("\n测试完成")
