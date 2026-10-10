"""模型类型识别和兼容性处理。

识别checkpoint中的模型类型，并自动使用对应的生成方式。
支持event_v1、event_ctx_v1及其扩展版本。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class ModelFamily(Enum):
    """模型族类型。"""
    EVENT_V1 = "event_v1"
    EVENT_CTX_V1 = "event_ctx_v1"
    EVENT_CTX_V1_EXTENDED = "event_ctx_v1_extended"  # 新增：支持节奏倾向和冲突检测


class FeatureSupport:
    """模型特性支持情况。"""
    
    # event_v1: 基础生成
    EVENT_V1_FEATURES = {
        "rhythm_bias": False,
        "fine_note_strength": False,
        "conflict_detection": False,
        "conflict_repair": False,
        "post_process": False,
    }
    
    # event_ctx_v1: 跨窗口上下文
    EVENT_CTX_V1_FEATURES = {
        "rhythm_bias": False,
        "fine_note_strength": False,
        "conflict_detection": False,
        "conflict_repair": False,
        "post_process": False,
    }
    
    # event_ctx_v1_extended: 支持新特性（无需重训，通过后处理）
    EVENT_CTX_V1_EXTENDED_FEATURES = {
        "rhythm_bias": True,  # 通过条件参数应用
        "fine_note_strength": True,
        "conflict_detection": True,
        "conflict_repair": True,
        "post_process": True,
    }
    
    @staticmethod
    def get_features(model_family: ModelFamily) -> dict[str, bool]:
        """获取模型的特性支持情况。"""
        if model_family == ModelFamily.EVENT_V1:
            return FeatureSupport.EVENT_V1_FEATURES.copy()
        elif model_family == ModelFamily.EVENT_CTX_V1:
            return FeatureSupport.EVENT_CTX_V1_FEATURES.copy()
        elif model_family == ModelFamily.EVENT_CTX_V1_EXTENDED:
            return FeatureSupport.EVENT_CTX_V1_EXTENDED_FEATURES.copy()
        else:
            # 默认：无新特性
            return FeatureSupport.EVENT_V1_FEATURES.copy()


def detect_model_family(checkpoint_data: dict) -> ModelFamily:
    """从checkpoint数据检测模型族。"""
    
    # 检查target_schema（优先）
    target_schema = checkpoint_data.get("data_contract", {}).get("target_schema")
    
    if target_schema == "maichart-event-ctx-v1":
        # 检查是否标记为扩展版本
        if checkpoint_data.get("data_contract", {}).get("rhythm_features_v1"):
            return ModelFamily.EVENT_CTX_V1_EXTENDED
        else:
            return ModelFamily.EVENT_CTX_V1
    
    elif target_schema == "maichart-event-v1":
        return ModelFamily.EVENT_V1
    
    # 备选：检查checkpoint中的其他标记
    checkpoint_type = checkpoint_data.get("checkpoint_type")
    if checkpoint_type == "event_ctx":
        if checkpoint_data.get("supports_rhythm_features"):
            return ModelFamily.EVENT_CTX_V1_EXTENDED
        return ModelFamily.EVENT_CTX_V1
    
    # 默认
    return ModelFamily.EVENT_V1


def validate_args_for_model(model_family: ModelFamily, args) -> tuple[bool, list[str]]:
    """验证CLI参数对于当前模型的有效性。
    
    返回：(是否有效, 警告信息列表)
    """
    features = FeatureSupport.get_features(model_family)
    warnings = []
    
    # 检查不支持的参数
    if hasattr(args, "rhythm_bias") and args.rhythm_bias != "auto" and not features["rhythm_bias"]:
        warnings.append(
            f"当前模型({model_family.value})不支持--rhythm-bias参数，将被忽略"
        )
    
    if hasattr(args, "fine_note_strength") and args.fine_note_strength != 0.5 and not features["fine_note_strength"]:
        warnings.append(
            f"当前模型({model_family.value})不支持--fine-note-strength参数，将被忽略"
        )
    
    if hasattr(args, "enable_conflict_detection") and args.enable_conflict_detection and not features["conflict_detection"]:
        warnings.append(
            f"当前模型({model_family.value})不支持冲突检测，--enable-conflict-detection将被忽略"
        )
    
    return len(warnings) == 0, warnings


def get_generation_config_for_model(
    model_family: ModelFamily,
    args
) -> dict:
    """为模型生成合适的配置字典。
    
    根据模型类型，决定哪些参数应该被使用。
    """
    config = {
        "model_family": model_family,
        "rhythm_bias": "auto",
        "fine_note_strength": 0.5,
        "enable_conflict_detection": False,
    }
    
    features = FeatureSupport.get_features(model_family)
    
    # 只有支持的参数才会被应用
    if features["rhythm_bias"] and hasattr(args, "rhythm_bias"):
        config["rhythm_bias"] = args.rhythm_bias
    
    if features["fine_note_strength"] and hasattr(args, "fine_note_strength"):
        config["fine_note_strength"] = args.fine_note_strength
    
    if features["conflict_detection"] and hasattr(args, "enable_conflict_detection"):
        config["enable_conflict_detection"] = args.enable_conflict_detection
        
        if hasattr(args, "detect_hard_conflicts"):
            config["detect_hard_conflicts"] = args.detect_hard_conflicts
        
        if hasattr(args, "detect_soft_conflicts"):
            config["detect_soft_conflicts"] = args.detect_soft_conflicts
        
        if hasattr(args, "conflict_repair_mode"):
            config["conflict_repair_mode"] = args.conflict_repair_mode
    
    return config


def mark_checkpoint_as_extended(checkpoint_data: dict) -> dict:
    """标记checkpoint为扩展版本（支持新特性）。
    
    用于在first-time使用后处理的checkpoint。
    """
    if "data_contract" not in checkpoint_data:
        checkpoint_data["data_contract"] = {}
    
    checkpoint_data["data_contract"]["rhythm_features_v1"] = True
    checkpoint_data["supports_rhythm_features"] = True
    
    return checkpoint_data


if __name__ == "__main__":
    print("=== 模型兼容性测试 ===\n")
    
    # 测试1: event_v1
    checkpoint_v1 = {
        "data_contract": {
            "target_schema": "maichart-event-v1"
        }
    }
    family = detect_model_family(checkpoint_v1)
    print(f"检测到模型: {family.value}")
    print(f"特性支持: {FeatureSupport.get_features(family)}\n")
    
    # 测试2: event_ctx_v1
    checkpoint_ctx = {
        "data_contract": {
            "target_schema": "maichart-event-ctx-v1"
        }
    }
    family = detect_model_family(checkpoint_ctx)
    print(f"检测到模型: {family.value}")
    print(f"特性支持: {FeatureSupport.get_features(family)}\n")
    
    # 测试3: event_ctx_v1_extended
    checkpoint_extended = {
        "data_contract": {
            "target_schema": "maichart-event-ctx-v1",
            "rhythm_features_v1": True
        }
    }
    family = detect_model_family(checkpoint_extended)
    print(f"检测到模型: {family.value}")
    print(f"特性支持: {FeatureSupport.get_features(family)}\n")
    
    print("兼容性测试完成")
