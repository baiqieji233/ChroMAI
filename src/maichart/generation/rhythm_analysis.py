"""细粒度节奏特征提取模块。

用于检测和量化复杂节奏模式（三连音、附点、多声部），
引导模型学习更积极的细碎音符采样。

这个模块不拼入模型输入（避免改变112维contract），
而是作为分析工具和数据标注辅助。
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


class RhythmAnalysis(NamedTuple):
    """细粒度节奏分析结果。"""
    
    # 节奏特征 [0, 1]
    triplet_strength: float      # 三连音倾向
    dotted_strength: float       # 附点节奏倾向
    syncopation: float           # 切分强度
    rhythm_entropy: float        # 节奏复杂度（归一化）
    
    # 配置类别
    dominant_rhythm: str         # "uniform", "triplet", "dotted", "mixed", "complex"
    fine_note_ratio: float       # [0, 1]，细碎音符占比
    
    # 多声部信息
    polyphony_score: float       # 声部独立性 [0, 1]
    num_voices: int              # 估计的声部数量


def analyze_window_rhythm(
    onsets: np.ndarray,
    onset_strength: np.ndarray,
    local_tempo_bpm: float,
    window_duration_sec: float = 4.0,
) -> RhythmAnalysis:
    """分析单个时间窗口的节奏特征。
    
    参数：
    - onsets: 窗口内onset事件的相对时间（秒）
    - onset_strength: 每个onset的能量 [0, 1]
    - local_tempo_bpm: 局部BPM
    - window_duration_sec: 窗口时长（秒）
    
    返回：RhythmAnalysis对象
    """
    if len(onsets) < 2:
        return RhythmAnalysis(
            triplet_strength=0.0,
            dotted_strength=0.0,
            syncopation=0.0,
            rhythm_entropy=0.0,
            dominant_rhythm="sparse",
            fine_note_ratio=0.0,
            polyphony_score=0.0,
            num_voices=1,
        )
    
    onsets = np.asarray(onsets, dtype=np.float32)
    onset_strength = np.asarray(onset_strength, dtype=np.float32)
    
    # 计算三连音强度
    triplet_strength = _compute_triplet_strength(onsets, onset_strength, local_tempo_bpm)
    
    # 计算附点节奏强度
    dotted_strength = _compute_dotted_strength(onsets, onset_strength)
    
    # 计算切分强度
    syncopation = _compute_syncopation(onsets, onset_strength, local_tempo_bpm)
    
    # 计算节奏熵
    rhythm_entropy = _compute_rhythm_entropy(onsets, window_duration_sec)
    
    # 判断主要节奏类型
    dominant = _classify_dominant_rhythm(
        triplet_strength, dotted_strength, syncopation, rhythm_entropy
    )
    
    # 计算细碎音符占比（12分/16分/24分音符）
    fine_ratio = _compute_fine_note_ratio(onsets, local_tempo_bpm, window_duration_sec)
    
    # 多声部检测（简化版）
    polyphony, num_voices = _estimate_polyphony(onsets, onset_strength)
    
    return RhythmAnalysis(
        triplet_strength=float(np.clip(triplet_strength, 0, 1)),
        dotted_strength=float(np.clip(dotted_strength, 0, 1)),
        syncopation=float(np.clip(syncopation, 0, 1)),
        rhythm_entropy=float(np.clip(rhythm_entropy / 4.0, 0, 1)),  # 归一化到[0,1]
        dominant_rhythm=dominant,
        fine_note_ratio=float(np.clip(fine_ratio, 0, 1)),
        polyphony_score=float(np.clip(polyphony, 0, 1)),
        num_voices=num_voices,
    )


def _compute_triplet_strength(
    onsets: np.ndarray, 
    onset_strength: np.ndarray, 
    local_tempo_bpm: float
) -> float:
    """计算三连音倾向。
    
    原理：计算1/3拍网格的onset能量峰值与1/2拍、1/4拍网格的对比。
    """
    if len(onsets) < 3:
        return 0.0
    
    beat_sec = 60.0 / max(local_tempo_bpm, 1.0)
    triplet_interval = beat_sec / 3.0  # 三连音间隔
    quarter_interval = beat_sec / 4.0  # 四分音符间隔
    
    # 计算onset在各网格上的对齐度
    triplet_score = _grid_alignment_score(onsets, onset_strength, triplet_interval)
    quarter_score = _grid_alignment_score(onsets, onset_strength, quarter_interval)
    uniform_score = _grid_alignment_score(onsets, onset_strength, beat_sec)
    
    # 三连音强度 = (triplet_score - quarter_score) 相对于baseline
    if quarter_score + uniform_score > 0:
        strength = max(0.0, (triplet_score - quarter_score) / (quarter_score + uniform_score + 1e-6))
    else:
        strength = 0.0
    
    return float(strength)


def _compute_dotted_strength(
    onsets: np.ndarray, 
    onset_strength: np.ndarray
) -> float:
    """计算附点节奏倾向。
    
    原理：检测3:1和5:1时长比的重复模式。
    """
    if len(onsets) < 3:
        return 0.0
    
    intervals = np.diff(onsets)
    if len(intervals) < 2:
        return 0.0
    
    # 寻找3:1模式 (长-短-长-短)
    dotted_count = 0
    total_pairs = 0
    
    for i in range(len(intervals) - 1):
        if intervals[i] > 0 and intervals[i+1] > 0:
            ratio = max(intervals[i], intervals[i+1]) / min(intervals[i], intervals[i+1])
            if 2.5 < ratio < 3.5:  # 3:1 ±0.5容差
                dotted_count += 1
            total_pairs += 1
    
    if total_pairs > 0:
        return float(dotted_count / total_pairs)
    return 0.0


def _compute_syncopation(
    onsets: np.ndarray, 
    onset_strength: np.ndarray, 
    local_tempo_bpm: float
) -> float:
    """计算切分强度。
    
    原理：弱拍的能量与强拍能量的比值。
    """
    if len(onsets) < 4:
        return 0.0
    
    beat_sec = 60.0 / max(local_tempo_bpm, 1.0)
    
    # 将onsets分类到强拍和弱拍
    strong_strength = []
    weak_strength = []
    
    for onset, strength in zip(onsets, onset_strength):
        beat_phase = (onset / beat_sec) % 4.0
        if abs(beat_phase) < 0.5 or abs(beat_phase - 2.0) < 0.5:
            strong_strength.append(strength)
        else:
            weak_strength.append(strength)
    
    if len(strong_strength) > 0 and len(weak_strength) > 0:
        strong_avg = np.mean(strong_strength)
        weak_avg = np.mean(weak_strength)
        if strong_avg > 0:
            syncopation = weak_avg / strong_avg
        else:
            syncopation = 0.0
    else:
        syncopation = 0.0
    
    return float(np.clip(syncopation, 0, 1))


def _compute_rhythm_entropy(onsets: np.ndarray, window_duration_sec: float) -> float:
    """计算节奏熵（音符间隔分布的多样性）。
    
    原理：使用onset间隔分布的熵度量节奏复杂度。
    高熵 = 复杂多样，低熵 = 单调重复。
    """
    if len(onsets) < 3:
        return 0.0
    
    intervals = np.diff(onsets)
    if len(intervals) == 0:
        return 0.0
    
    # 将间隔分成10个bin
    min_interval = np.min(intervals)
    max_interval = np.max(intervals)
    
    if max_interval - min_interval < 1e-6:
        return 0.0
    
    bins = np.linspace(min_interval, max_interval, 11)
    hist, _ = np.histogram(intervals, bins=bins)
    
    # 计算熵
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return float(entropy)


def _compute_fine_note_ratio(
    onsets: np.ndarray,
    local_tempo_bpm: float,
    window_duration_sec: float
) -> float:
    """计算细碎音符占比（12分/16分/24分音符）。
    
    原理：统计音符间隔中，小于1/4拍的比例。
    """
    if len(onsets) < 2:
        return 0.0
    
    beat_sec = 60.0 / max(local_tempo_bpm, 1.0)
    quarter_beat = beat_sec / 4.0
    
    intervals = np.diff(onsets)
    fine_intervals = np.sum(intervals < quarter_beat * 0.7)  # 细碎阈值
    
    if len(intervals) > 0:
        return float(fine_intervals / len(intervals))
    return 0.0


def _estimate_polyphony(
    onsets: np.ndarray, 
    onset_strength: np.ndarray
) -> tuple[float, int]:
    """估计多声部复杂度（简化版）。
    
    原理：基于onset密度和强度分布估计。
    """
    if len(onsets) < 2:
        return 0.0, 1
    
    intervals = np.diff(onsets)
    onset_density = len(onsets) / (onsets[-1] - onsets[0]) if onsets[-1] > onsets[0] else 0
    
    # 粗略估计：高密度+多层级强度 → 多声部
    strength_std = float(np.std(onset_strength))
    polyphony_score = min(1.0, onset_density * 2.0 + strength_std)
    
    # 估计声部数量（1-3）
    num_voices = 1 + int(polyphony_score / 0.4)
    
    return float(polyphony_score), num_voices


def _grid_alignment_score(
    onsets: np.ndarray, 
    onset_strength: np.ndarray, 
    grid_interval: float,
    tolerance: float = 0.05
) -> float:
    """计算onset在指定网格上的对齐度。"""
    scores = []
    for onset, strength in zip(onsets, onset_strength):
        phase = (onset % grid_interval) / grid_interval
        # 计算到最近网格点的距离
        dist_to_grid = min(phase, 1.0 - phase)
        if dist_to_grid < tolerance:
            scores.append(strength * (1.0 - dist_to_grid / tolerance))
        else:
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def _classify_dominant_rhythm(
    triplet_strength: float,
    dotted_strength: float,
    syncopation: float,
    rhythm_entropy: float
) -> str:
    """基于特征分类主要节奏类型。"""
    if triplet_strength > 0.6:
        return "triplet"
    elif dotted_strength > 0.5:
        return "dotted"
    elif rhythm_entropy > 2.0:
        return "complex"
    elif syncopation > 0.6:
        return "syncopated"
    else:
        return "uniform"


if __name__ == "__main__":
    # 简单测试
    print("=== 节奏分析测试 ===")
    
    # 测试1: 均匀节拍
    onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    strength = np.ones_like(onsets)
    result = analyze_window_rhythm(onsets, strength, 120.0)
    print(f"\n均匀节拍: {result.dominant_rhythm} (triplet={result.triplet_strength:.2f})")
    
    # 测试2: 三连音
    onsets = np.array([0.0, 1/3, 2/3, 1.0, 4/3, 5/3])
    strength = np.ones_like(onsets)
    result = analyze_window_rhythm(onsets, strength, 120.0)
    print(f"三连音: {result.dominant_rhythm} (triplet={result.triplet_strength:.2f})")
    
    # 测试3: 附点
    onsets = np.array([0.0, 0.75, 1.0, 1.75, 2.0])
    strength = np.ones_like(onsets)
    result = analyze_window_rhythm(onsets, strength, 120.0)
    print(f"附点: {result.dominant_rhythm} (dotted={result.dotted_strength:.2f})")
    
    print("\n测试完成")
