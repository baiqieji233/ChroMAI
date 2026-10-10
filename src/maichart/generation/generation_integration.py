"""WebUI API端点扩展 - 节奏倾向和冲突检测参数。

这些函数应该被集成到webui/server.py中的生成API处理中。
"""

from __future__ import annotations

from typing import Any, Callable


class RhythmBiasHandler:
    """处理节奏倾向参数的handler。"""
    
    # 映射rhythm_bias到实际的条件参数调整
    BIAS_PRESETS = {
        "auto": {
            "description": "自动（跟随音乐）",
            "fine_note_strength": 0.5,
            "apply_rhythm_conditions": False
        },
        "uniform": {
            "description": "均匀节拍",
            "fine_note_strength": 0.2,
            "note_density_adjustment": 1.0,
            "slide_bias_adjustment": -0.3  # 减少复杂节奏
        },
        "triplet": {
            "description": "三连音倾向",
            "fine_note_strength": 0.8,
            "rhythm_hint": "triplet"
        },
        "dotted": {
            "description": "附点节奏",
            "fine_note_strength": 0.7,
            "rhythm_hint": "dotted"
        },
        "mixed": {
            "description": "混合复杂",
            "fine_note_strength": 0.9,
            "rhythm_hint": "complex"
        }
    }
    
    @staticmethod
    def apply_rhythm_bias(
        base_conditions: dict[str, float],
        rhythm_bias: str,
        fine_note_strength: float
    ) -> dict[str, float]:
        """应用节奏倾向到条件参数。
        
        参数：
        - base_conditions: 基础条件字典（来自radar/style）
        - rhythm_bias: 节奏倾向 ("auto", "uniform", "triplet", "dotted", "mixed")
        - fine_note_strength: 细碎音符强度 [0, 1]
        
        返回：调整后的条件字典
        """
        if rhythm_bias not in RhythmBiasHandler.BIAS_PRESETS:
            rhythm_bias = "auto"
        
        adjusted = dict(base_conditions)
        preset = RhythmBiasHandler.BIAS_PRESETS[rhythm_bias]
        
        # 应用节奏特定的条件调整
        if "note_density_adjustment" in preset:
            if "note_density" in adjusted:
                adjusted["note_density"] *= preset["note_density_adjustment"]
        
        if "slide_bias_adjustment" in preset:
            if "slide_bias" in adjusted:
                adjusted["slide_bias"] *= preset["slide_bias_adjustment"]
        
        # 细碎音符强度可以通过pattern_reuse间接控制
        # （保守采音时pattern_reuse降低，积极采音时提高）
        if fine_note_strength < 0.3:
            adjusted["pattern_reuse"] = adjusted.get("pattern_reuse", 0.028) * 0.5
        elif fine_note_strength > 0.7:
            adjusted["pattern_reuse"] = adjusted.get("pattern_reuse", 0.028) * 1.5
        
        return adjusted


class ConflictDetectionHandler:
    """处理冲突检测参数的handler。"""
    
    @staticmethod
    def validate_config(config: dict[str, Any]) -> tuple[bool, str]:
        """验证冲突检测配置。"""
        if not isinstance(config.get("enable_conflict_detection", True), bool):
            return False, "enable_conflict_detection 必须是布尔值"
        
        repair_mode = config.get("conflict_repair_mode", "none")
        if repair_mode not in ("none", "local", "reject"):
            return False, f"无效的冲突处理模式: {repair_mode}"
        
        return True, ""
    
    @staticmethod
    def get_detector_params(config: dict[str, Any]) -> dict[str, Any]:
        """从配置中提取冲突检测器参数。"""
        return {
            "enable_hard_conflicts": config.get("detect_hard_conflicts", True),
            "enable_soft_conflicts": config.get("detect_soft_conflicts", False),
            "max_simultaneous_buttons": config.get("max_simultaneous_buttons", 4),
            "min_tap_interval": config.get("min_tap_interval", 0.05),
            "max_jump_distance": config.get("max_jump_distance", 1.8),
            "min_jump_time": config.get("min_jump_time", 0.12)
        }


def integrate_generation_config(
    base_config: dict[str, Any],
    generation_params: dict[str, Any]
) -> dict[str, Any]:
    """整合新的节奏和冲突检测参数到生成配置中。
    
    参数：
    - base_config: 原始生成配置
    - generation_params: 新的节奏和冲突检测参数
    
    返回：整合后的配置
    """
    result = dict(base_config)
    
    # 提取节奏参数
    rhythm_bias = generation_params.get("rhythm_bias", "auto")
    fine_note_strength = generation_params.get("fine_note_strength", 0.5)
    
    # 应用节奏倾向到条件
    if "conditions" in result:
        result["conditions"] = RhythmBiasHandler.apply_rhythm_bias(
            result["conditions"],
            rhythm_bias,
            fine_note_strength
        )
    
    # 添加冲突检测配置
    result["post_process_config"] = {
        "enable_conflict_detection": generation_params.get("enable_conflict_detection", True),
        "detect_hard_conflicts": generation_params.get("detect_hard_conflicts", True),
        "detect_soft_conflicts": generation_params.get("detect_soft_conflicts", False),
        "conflict_repair_mode": generation_params.get("conflict_repair_mode", "none"),
    }
    
    # 验证冲突检测配置
    valid, msg = ConflictDetectionHandler.validate_config(result["post_process_config"])
    if not valid:
        raise ValueError(f"无效的冲突检测配置: {msg}")
    
    return result


# WebUI集成示例代码（应添加到webui/server.py）

WEBUI_API_HANDLER_EXAMPLE = """
# 在webui/server.py的_start_generate方法中添加以下代码

def _start_generate(self, payload: dict[str, Any]) -> None:
    # ... 原有代码 ...
    
    # 新增：处理节奏和冲突检测参数
    generation_params = {
        "rhythm_bias": payload.get("rhythm_bias", "auto"),
        "fine_note_strength": payload.get("fine_note_strength", 0.5),
        "enable_conflict_detection": payload.get("enable_conflict_detection", True),
        "detect_hard_conflicts": payload.get("detect_hard_conflicts", True),
        "detect_soft_conflicts": payload.get("detect_soft_conflicts", False),
        "conflict_repair_mode": payload.get("conflict_repair_mode", "none"),
    }
    
    # 整合配置
    config = integrate_generation_config(base_generation_config, generation_params)
    
    # ... 生成流程 ...
    # 生成后运行后处理
    from maichart.generation.post_process import run_post_process, PostProcessConfig
    
    post_process_config = PostProcessConfig(**config["post_process_config"])
    result = run_post_process(notes_dict, post_process_config, verbose=False)
    
    if not result.success and config["post_process_config"]["conflict_repair_mode"] == "reject":
        raise RuntimeError(f"生成被冲突检测拒绝: {result.message}")
    
    # 保存报告
    if result.conflicts_detected or result.repairs_applied:
        from maichart.generation.post_process import save_post_process_report
        save_post_process_report(result, f"{output_dir}/post_process_report.json")
"""


if __name__ == "__main__":
    print("=== WebUI集成测试 ===")
    
    # 测试1: 应用节奏倾向
    base_cond = {
        "note_density": 0.305,
        "slide_bias": 0.140,
        "pattern_reuse": 0.028
    }
    
    adjusted = RhythmBiasHandler.apply_rhythm_bias(base_cond, "triplet", 0.8)
    print(f"\n节奏倾向应用 (triplet):")
    print(f"  slide_bias: {base_cond['slide_bias']} -> {adjusted.get('slide_bias', 'unchanged')}")
    print(f"  pattern_reuse: {base_cond['pattern_reuse']} -> {adjusted.get('pattern_reuse')}")
    
    # 测试2: 验证配置
    config = {
        "enable_conflict_detection": True,
        "detect_hard_conflicts": True,
        "conflict_repair_mode": "local"
    }
    valid, msg = ConflictDetectionHandler.validate_config(config)
    print(f"\n配置验证: {'✓' if valid else '✗'} {msg}")
    
    # 测试3: 提取检测器参数
    params = ConflictDetectionHandler.get_detector_params(config)
    print(f"\n检测器参数数量: {len(params)}")
    
    print("\n集成完成")
