"""训练集节奏标注脚本。

为训练集中的每个谱面窗口添加节奏特征标注，
用于引导模型学习复杂节奏采样。

使用方法：
python scripts/annotate_rhythm_patterns.py \
    --dataset webui_workspace/datasets/full训练集 \
    --output webui_workspace/datasets/full训练集/rhythm_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from maichart.generation.rhythm_analysis import analyze_window_rhythm, RhythmAnalysis
from maichart.training.event_dataset import EventDataset


def annotate_dataset_rhythm(
    dataset_path: str,
    output_path: str | None = None,
    window_size_beats: int = 16,
    verbose: bool = True
) -> dict[str, Any]:
    """为整个数据集标注节奏特征。
    
    参数：
    - dataset_path: 数据集manifest路径
    - output_path: 输出标注结果的路径（默认覆盖原manifest）
    - window_size_beats: 窗口大小（拍数）
    - verbose: 打印进度信息
    
    返回：标注后的manifest字典
    """
    if verbose:
        print(f"加载数据集: {dataset_path}")
    
    # 读取manifest
    with open(dataset_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    if "segments" not in manifest:
        print("警告: manifest中没有segments字段")
        return manifest
    
    segments = manifest["segments"]
    annotated_count = 0
    
    for seg_idx, segment in enumerate(segments):
        if verbose and seg_idx % 100 == 0:
            print(f"处理进度: {seg_idx}/{len(segments)}")
        
        # 提取窗口内的音符时间和强度
        try:
            rhythm = _extract_rhythm_from_segment(segment, window_size_beats)
            
            if rhythm:
                # 添加节奏标注到segment
                segment["rhythm_analysis"] = {
                    "triplet_strength": rhythm.triplet_strength,
                    "dotted_strength": rhythm.dotted_strength,
                    "syncopation": rhythm.syncopation,
                    "rhythm_entropy": rhythm.rhythm_entropy,
                    "dominant_rhythm": rhythm.dominant_rhythm,
                    "fine_note_ratio": rhythm.fine_note_ratio,
                    "polyphony_score": rhythm.polyphony_score,
                    "num_voices": rhythm.num_voices,
                }
                annotated_count += 1
        except Exception as e:
            if verbose:
                print(f"警告: segment {seg_idx} 标注失败: {e}")
            continue
    
    if verbose:
        print(f"\n标注完成: {annotated_count}/{len(segments)} 个segments")
    
    # 保存标注后的manifest
    if output_path is None:
        output_path = dataset_path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"保存到: {output_path}")
    
    return manifest


def _extract_rhythm_from_segment(segment: dict[str, Any], window_size_beats: int) -> RhythmAnalysis | None:
    """从segment中提取节奏特征。"""
    # 获取窗口内的事件
    if "events" not in segment:
        return None
    
    events = segment.get("events", [])
    if not events:
        return None
    
    # 提取音符时间（相对于窗口起点）
    note_times = []
    note_strengths = []
    
    for event in events:
        if "beat" in event:
            # 转换beat到秒（假设BPM）
            bpm = segment.get("local_bpm", 120.0)
            time_sec = float(event["beat"]) * 60.0 / bpm
            note_times.append(time_sec)
            
            # 简化：所有音符强度设为1.0
            # 实际可以根据note类型（Tap/Hold/Slide）调整
            note_strengths.append(1.0)
    
    if len(note_times) < 2:
        return None
    
    onsets = np.array(note_times, dtype=np.float32)
    strengths = np.array(note_strengths, dtype=np.float32)
    
    # 计算窗口时长
    bpm = segment.get("local_bpm", 120.0)
    window_duration_sec = window_size_beats * 60.0 / bpm
    
    # 分析节奏
    return analyze_window_rhythm(onsets, strengths, bpm, window_duration_sec)


def print_dataset_rhythm_summary(manifest: dict[str, Any]) -> None:
    """打印数据集节奏特征统计摘要。"""
    segments = manifest.get("segments", [])
    
    if not segments:
        print("数据集为空")
        return
    
    # 统计各类节奏模式的数量
    rhythm_counts = {}
    triplet_scores = []
    dotted_scores = []
    fine_ratios = []
    
    for segment in segments:
        rhythm = segment.get("rhythm_analysis")
        if rhythm:
            dominant = rhythm.get("dominant_rhythm", "unknown")
            rhythm_counts[dominant] = rhythm_counts.get(dominant, 0) + 1
            
            triplet_scores.append(rhythm.get("triplet_strength", 0.0))
            dotted_scores.append(rhythm.get("dotted_strength", 0.0))
            fine_ratios.append(rhythm.get("fine_note_ratio", 0.0))
    
    print("\n=== 数据集节奏特征摘要 ===")
    print(f"总segments数: {len(segments)}")
    print(f"已标注: {len(triplet_scores)}")
    
    print("\n节奏模式分布:")
    for rhythm_type, count in sorted(rhythm_counts.items(), key=lambda x: -x[1]):
        percent = 100.0 * count / len(triplet_scores) if triplet_scores else 0
        print(f"  {rhythm_type:12s}: {count:5d} ({percent:5.1f}%)")
    
    if triplet_scores:
        print(f"\n三连音强度: 平均 {np.mean(triplet_scores):.3f}, 中位数 {np.median(triplet_scores):.3f}")
        print(f"附点强度:   平均 {np.mean(dotted_scores):.3f}, 中位数 {np.median(dotted_scores):.3f}")
        print(f"细碎音符比: 平均 {np.mean(fine_ratios):.3f}, 中位数 {np.median(fine_ratios):.3f}")
        
        # 统计高复杂度窗口
        high_triplet = sum(1 for s in triplet_scores if s > 0.6)
        high_dotted = sum(1 for s in dotted_scores if s > 0.5)
        high_fine = sum(1 for s in fine_ratios if s > 0.3)
        
        print(f"\n高复杂度窗口:")
        print(f"  三连音倾向 >0.6: {high_triplet} ({100*high_triplet/len(triplet_scores):.1f}%)")
        print(f"  附点倾向 >0.5:   {high_dotted} ({100*high_dotted/len(triplet_scores):.1f}%)")
        print(f"  细碎音符 >0.3:   {high_fine} ({100*high_fine/len(triplet_scores):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="为训练集标注节奏特征")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集manifest.json路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--window-beats",
        type=int,
        default=16,
        help="窗口大小（拍数），默认16"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="显示统计摘要"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式"
    )
    
    args = parser.parse_args()
    
    # 执行标注
    manifest = annotate_dataset_rhythm(
        dataset_path=args.dataset,
        output_path=args.output,
        window_size_beats=args.window_beats,
        verbose=not args.quiet
    )
    
    # 显示摘要
    if args.summary:
        print_dataset_rhythm_summary(manifest)


if __name__ == "__main__":
    main()
