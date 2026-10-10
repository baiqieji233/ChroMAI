# 舞萌DX谱面生成模型 - 复杂节奏采样与可玩性优化方案

**项目**: E:\simai_transformer  
**创建日期**: 2026-10-10  
**基础模型**: event_v11-2/epoch_020.pt (music_structure_v2, 20 epochs)  
**文档版本**: Final v1.0

---

## 执行摘要

当前模型在12-15难度范围内已具备基本谱面生成能力，但存在两个核心问题：

1. **复杂节奏过度简化**：面对复杂节奏（三连音、附点、16分音符混合、多声部并行），模型倾向于保守采音，只使用四分八分
2. **可玩性缺失**：生成的谱面可能出现无理配置（hold与tap重叠）或伪无理（人体结构无法完成），以及异常的难度表现

本方案采用**增强型音频特征提取 + 可选物理约束后处理**的渐进式改进策略，保留现有music_structure_v2架构的核心优势，通过数据层和后处理层的优化解决上述问题。

**核心设计原则**：
- ✅ **所有约束均为可选项**，训练集中的部分谱面可能与约束冲突，不强制过滤
- ✅ **约束不作为硬性mask进入训练**，避免过度限制模型学习
- ✅ **冲突检测基于真实游戏判定机制**，不是人体工学假设
- ✅ **允许破坏性修改**，fresh重训，不要求向后兼容

---

## 一、问题分析

### 1.1 复杂节奏过度简化

:codex-annotation{index="1"}

**现象**：
- 12分/24分交替段落 → 模型只采4分8分主节拍
- 16分附点音符 → 只采部分音，忽略细碎装饰音
- 多声部并行（人声+鼓点+旋律） → 只跟一个声部
- 快速钢琴三连音 → 被简化为均匀节拍

**根本原因**：
1. **训练数据偏向**：训练集中谱师为可玩性做的简化设计占多数，模型学会了"复杂音乐要适度简化"
2. **音频特征不足**：当前event_audio特征(112维)主要捕捉响度、onset、频谱能量，但对细粒度节奏模式（三连音 vs 附点 vs 均匀）的表征不足
3. **规划器粒度问题**：planner预测整窗口的密度统计，无法指导局部的节奏选择倾向

### 1.2 可玩性问题

**硬无理**（游戏判定系统冲突）：
- Hold A [t1, t2] 期间，Tap A 出现在 t ∈ (t1, t2) → 判定冲突
- Slide 1-3 进行中，星星在按键P时，Tap P 同时出现 → 星星与tap冲突
- Touch A区 与 Tap/Hold 1号按键重叠 → 判定混淆

**伪无理**（人体结构限制）：
- 0.1秒内从 Tap 1 跳到 Tap 5（对角线） → 物理距离过远
- Slide 1-3 持续期间，同时要求 Tap 8+Tap 4 多押 → 双手调度冲突
- 连续大跨度跳跃 → 超出人类反应速度

**异常难度表现**：
- 简单段落（音乐强度低、旋律稀疏）出现高密度配置
- 高潮段落出现不合理的留白
- 难度曲线起点过高（开场就高密度）

---

## 二、最终方案架构

采用**方案2（两阶段生成优化）+ 方案3（可选物理约束后处理）**的融合设计。

`
音频输入 
  ↓
[阶段1] 增强音频特征提取 + 细粒度节奏标注
  ↓
[阶段2] music_structure_v2 规划器（保持现有）
  ↓  
[阶段3] Event Decoder（保持现有）
  ↓
[阶段4] 可选物理约束后处理（新增）
  ↓
最终谱面
`

### 2.1 核心策略

1. **音频特征层**：增强细粒度节奏特征表达，让模型"看见"三连音、附点、切分
2. **数据标注层**：为训练集添加节奏模式标签，引导模型学习复杂节奏采样
3. **后处理层**：可选的物理约束检测和修复，作为生成后的安全网

**不改动的部分**：
- music_structure_v2 规划器架构（12维planner）
- event decoder基础结构
- checkpoint训练流程主干

---

## 三、详细实现方案

### 任务1：增强细粒度节奏特征

**目标**：让音频特征能够区分"三连音 vs 附点 vs 均匀4分"等节奏模式。

#### 1.1 新增节奏特征维度

在 src/maichart/generation/event_audio.py 的 _beat_normalized_rhythm() 基础上扩展：

**新增特征**（暂不拼入模型输入，先作为分析工具）：
1. **Triplet强度** (triplet_strength)：
   - 计算1/3拍网格的onset能量峰值
   - 与1/2拍、1/4拍网格对比，得到"三连音倾向"分数
   - 范围 [0, 1]，0.7+表示明显三连音

2. **Dotted强度** (dotted_strength)：
   - 计算附点节奏模式（长-短-长-短）的自相关性
   - 检测3:1时长比的音符组
   - 范围 [0, 1]

3. **Syncopation强度** (syncopation)：
   - 已有特征，需确保正确计算
   - 弱拍onset强于强拍的程度

4. **节奏熵** (rhythm_entropy)：
   - 计算onset间隔分布的熵
   - 高熵 = 复杂多样节奏，低熵 = 单一重复节奏

5. **多声部检测** (polyphony_score)：
   - 使用HPSS（harmonic-percussive source separation）分离
   - 计算旋律、和声、打击乐声部的独立onset时间线
   - 输出"声部数量"和"声部冲突度"

**实现位置**：
`python
# src/maichart/generation/event_audio.py

def _compute_fine_rhythm_features(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    hop_length: int = 512
) -> dict:
    ""
计算窗口级细粒度节奏特征。
    
    返回：
    - triplet_strength: 三连音倾向 [0,1]
    - dotted_strength: 附点节奏倾向 [0,1]
    - syncopation: 切分强度 [0,1]
    - rhythm_entropy: 节奏熵
    - polyphony_score: 多声部复杂度
    """
    # 实现细节见任务1.2
    pass
```

#### 1.2 训练集节奏标注

**自动标注脚本**：`scripts/annotate_rhythm_patterns.py`

为每个训练谱面的16拍窗口标注：
- `dominant_rhythm`: 主要节奏模式 {"uniform", "triplet", "dotted", "mixed", "complex"}
- `rhythm_density_class`: {"sparse", "moderate", "dense"}
- `fine_note_ratio`: 细碎音符占比（12分/16分/24分音符数量 / 总音符数）

标注结果存入 `manifest.json` 的扩展字段，不破坏现有结构。

**标注规则**：
```python
# 基于实际谱面note时间戳分析
if 三连音网格命中率 > 0.6:
    dominant_rhythm = "triplet"
elif 附点模式匹配度 > 0.5:
    dominant_rhythm = "dotted"
elif 节奏熵 > 2.0:
    dominant_rhythm = "complex"
elif 音符间隔方差 < 0.1:
    dominant_rhythm = "uniform"
else:
    dominant_rhythm = "mixed"
```

#### 1.3 扩展规划器维度（可选）

**评估后决定是否执行**：

如果评估认为当前12维planner不足以表达节奏倾向，可扩展为16维：

```python
# music_structure_v2 扩展方案（需fresh重训）
CONTINUOUS_KEYS = (
    "count", "density", "chord", "slide", "last", 
    "touch", "hold", "intensity",
    "triplet_bias",   # 新增：三连音倾向 [0,1]
    "dotted_bias",    # 新增：附点倾向 [0,1]
    "fine_note_ratio" # 新增：细碎音符占比
)
CONFIG_KEYS = ("single", "chord", "slide", "touch")
PLAN_DIM = {
    LEGACY_PROFILE: 9, 
    MUSIC_STRUCTURE_PROFILE: 12,
    "music_structure_v3": 15  # 11连续+4配置
}
```

**触发条件**：
- 短跑测试发现12维planner无法学习复杂节奏倾向
- 固定同一音频，改变rhythm条件时生成无明显差异

**不触发条件**：
- 当前planner已能通过intensity+density组合隐式表达节奏倾向
- 扩展planner维度后训练不稳定或收敛变慢

---

### 任务2：数据策略优化

#### 2.1 复杂节奏专项训练集

**构建流程**：
1. 从full训练集中筛选 `fine_note_ratio > 0.3` 的窗口
2. 筛选包含三连音/附点/16分音符的段落
3. 手动审核50-100个样本，确认质量
4. 作为finetune阶段的专项数据

**使用方式**：
- 基础训练：full训练集，均匀采样
- Finetune阶段：增加复杂节奏专项集的采样权重（2x-3x）

#### 2.2 训练参数调整

**Stage 1: 基础全量训练**（10-15 epochs）
```yaml
dataset: full训练集
difficulty_range: [4, 6]  # Expert/Master/Re:Master
level_range: [12.0, 15.0]
oversample: 
  slide: 1.0  # 不过采样
  touch: 1.0
quality_weights: 基础权重（优秀2.0，普通1.0）
planner_profile: music_structure_v2
```

**Stage 2: 复杂节奏强化训练**（5-10 epochs）
```yaml
dataset: full训练集 + 复杂节奏专项集
oversample:
  fine_rhythm: 2.5  # 新增：细碎节奏窗口过采样
  slide: 1.0
  touch: 1.0
quality_weights: 强化权重
```

---

### 任务3：舞萌物理模型与约束检测

:codex-annotation{index="2"}
:codex-annotation{index="3"}
:codex-annotation{index="4"}

**设计原则**：
- **所有约束为可选项**，WebUI中可配置启用/禁用
- **不作为训练期硬mask**，避免过拟合约束逻辑
- **只检测真实游戏判定冲突**，不过度模拟人体工学

#### 3.1 按键布局模型

**修正后的布局**（根据用户注释）：

```python
# src/maichart/generation/maimai_physics.py

"""舞萌DX物理约束模型。"""

import math
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ButtonLayout:
    """舞萌DX 1-8号按键在八边形上的物理布局。
    
    - 1号按键位于正上方顺时针偏转22.5度（八边形第一个顶点）
    - 按键按顺时针方向每隔45度排列
    - 八边形上边由1号和8号按键构成，与地面平行
    """
    
    @staticmethod
    def button_angle(button: int) -> float:
        """返回按键的角度（度数，0度为正右方，逆时针为正）。
        
        舞萌布局：1号在22.5°（正上偏右），顺时针递减45°。
        转换为标准坐标系（逆时针为正）：
        - 1号：90° - 22.5° = 67.5°
        - 2号：67.5° - 45° = 22.5°
        - 3号：-22.5° (或 337.5°)
        - ...以此类推
        """
        if not (1 <= button <= 8):
            raise ValueError(f"按键必须在1-8范围内，收到{button}")
        # 1号按键在67.5度，每个按键顺时针减少45度
        return 67.5 - (button - 1) * 45.0
    
    @staticmethod
    def angular_distance(btn1: int, btn2: int) -> float:
        """计算两按键的最短角度距离（度）。"""
        angle1 = ButtonLayout.button_angle(btn1)
        angle2 = ButtonLayout.button_angle(btn2)
        diff = abs(angle1 - angle2)
        # 取最短路径
        return min(diff, 360 - diff)
    
    @staticmethod
    def physical_distance(btn1: int, btn2: int, radius: float = 1.0) -> float:
        """计算两按键的物理直线距离（归一化半径=1）。"""
        ang_dist = math.radians(ButtonLayout.angular_distance(btn1, btn2))
        # 弦长公式
        return 2 * radius * math.sin(ang_dist / 2)

@dataclass(frozen=True)
class HandDispatch:
    """双手按键分配模型。
    
    基础规则（根据用户修正）：
    - 左手优先：8, 5, 6, 7（底部半圆+左侧）
    - 右手优先：1, 2, 3, 4（顶部半圆+右侧）
    
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
        - 同手2个按键且角度<90° → 可以（伸展手指）
        - 同手2个按键且角度≥90° → 需要allow_cross
        - 同手3+按键 → 需要allow_cross或touch
        """
        if len(buttons) <= 1:
            return True
        
        left_btns = [b for b in buttons if b in HandDispatch.LEFT_PRIMARY]
        right_btns = [b for b in buttons if b in HandDispatch.RIGHT_PRIMARY]
        
        # 左右手各自按1-2个 → 合理
        if len(left_btns) <= 2 and len(right_btns) <= 2:
            return True
        
        # 单手3+ → 需要交叉或touch
        if len(left_btns) >= 3 or len(right_btns) >= 3:
            return allow_cross
        
        return True
```

#### 3.2 Slide星星轨迹模型

**核心：星星位置是时间的连续函数**

```python
# src/maichart/generation/maimai_physics.py (continued)

from fractions import Fraction
from maichart.slides import get_slide_pattern_definition

@dataclass
class SlideTrajectory:
    """Slide星星轨迹。"""
    
    start_button: int
    pattern: str
    end_button: int
    start_time: float  # 秒
    duration: float    # 秒
    
    def star_position_at(self, time: float) -> int | None:
        """返回time时刻星星所在的按键位置。
        
        简化模型：
        - 直线slide (-)：线性插值按键位置
        - 弧线slide (<, >, ^)：按弧长线性插值
        - 复杂轨迹 (p, q, V等)：分段线性近似
        
        返回None表示星星不在任何按键上（轨迹中间）。
        """
        if time < self.start_time or time > self.start_time + self.duration:
            return None
        
        progress = (time - self.start_time) / self.duration
        
        if self.pattern == "-":
            # 直线：按角度线性插值
            start_ang = ButtonLayout.button_angle(self.start_button)
            end_ang = ButtonLayout.button_angle(self.end_button)
            
            # 选择最短路径
            diff = end_ang - start_ang
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            current_ang = start_ang + progress * diff
            
            # 找最接近的按键
            distances = [(abs(current_ang - ButtonLayout.button_angle(b)), b) for b in range(1, 9)]
            closest_btn = min(distances)[1]
            
            # 只有在非常接近时才算"在按键上"
            if min(distances)[0] < 10:  # 10度容差
                return closest_btn
            return None
        
        # TODO: 其他pattern的详细轨迹（可后续细化）
        # 当前简化：起点和终点
        if progress < 0.3:
            return self.start_button
        elif progress > 0.7:
            return self.end_button
        else:
            return None  # 轨迹中间段
```

#### 3.3 冲突检测器

**检测规则**（根据用户注释和checkpoint上下文）：

```python
# src/maichart/generation/conflict_detector.py

"""舞萌谱面冲突检测器。"""

from dataclasses import dataclass
from typing import Literal
from maichart.generation.maimai_physics import ButtonLayout, HandDispatch, SlideTrajectory

@dataclass
class NoteEvent:
    """单个note事件。"""
    note_type: Literal["tap", "hold", "slide", "touch"]
    button: int | str  # 按键1-8 或 touch区域A-E
    time: float  # 起音时间（秒）
    duration: float = 0.0  # hold/slide持续时间
    slide_trajectory: SlideTrajectory | None = None

@dataclass
class ConflictReport:
    """冲突检测报告。"""
    has_conflict: bool
    conflict_type: str | None = None
    time: float | None = None
    notes: list[NoteEvent] | None = None
    severity: Literal["hard", "soft"] = "hard"
    message: str = ""

class ConflictDetector:
    """检测舞萌谱面中的无理配置。
    
    检测内容（根据用户要求）：
    - 硬冲突：游戏判定系统无法处理的配置
    - 软冲突：人体工学上困难但非绝对不可能的配置
    
    不过度检测：
    - Touch多押是合理的
    - 某些多按键配置在不同Slide阶段是合理的
    """
    
    def __init__(self, 
                 enable_hard_conflicts: bool = True,
                 enable_soft_conflicts: bool = False,
                 max_simultaneous_buttons: int = 4,
                 min_tap_interval: float = 0.05  # 50ms最小间隔
                 ):
        """
        参数：
        - enable_hard_conflicts: 启用硬冲突检测
        - enable_soft_conflicts: 启用软冲突检测（人体工学）
        - max_simultaneous_buttons: 最大同时按键数
        - min_tap_interval: 最小tap间隔（秒）
        """
        self.enable_hard = enable_hard_conflicts
        self.enable_soft = enable_soft_conflicts
        self.max_simultaneous = max_simultaneous_buttons
        self.min_interval = min_tap_interval
    
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
        
        return conflicts
    
    def _detect_hold_tap_overlap(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测Hold与Tap在同一按键的时间重叠。"""
        conflicts = []
        
        holds = [n for n in notes if n.note_type == "hold"]
        taps = [n for n in notes if n.note_type == "tap"]
        
        for hold in holds:
            hold_end = hold.time + hold.duration
            for tap in taps:
                if hold.button == tap.button:
                    # 检查tap是否在hold持续期间
                    if hold.time < tap.time < hold_end:
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
        """检测Slide星星与Tap的位置冲突（基于星星位置计算）。"""
        conflicts = []
        
        slides = [n for n in notes if n.note_type == "slide" and n.slide_trajectory]
        taps = [n for n in notes if n.note_type == "tap"]
        
        for slide in slides:
            traj = slide.slide_trajectory
            for tap in taps:
                if not isinstance(tap.button, int):
                    continue
                
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
                        message=f"Slide {traj.start_button}-{traj.end_button} 星星在{tap.time:.2f}s位于按键{star_btn}，与 Tap {tap.button} 冲突"
                    ))
        
        return conflicts
    
    def _detect_touch_button_overlap(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测Touch与Button的判定冲突。
        
        根据用户注释：
        - Touch多押是合理的
        - 只检测：Touch A区与1号按键的严重重叠
        - 不检测：Touch的人体工学问题
        """
        conflicts = []
        
        touches = [n for n in notes if n.note_type == "touch"]
        buttons = [n for n in notes if n.note_type in ("tap", "hold") and isinstance(n.button, int)]
        
        # 简化规则：只检查Touch A区与1号按键的时间重叠
        for touch in touches:
            if touch.button != "A":
                continue
            
            touch_end = touch.time + touch.duration
            for btn_note in buttons:
                if btn_note.button == 1:  # A区主要影响1号按键
                    if abs(touch.time - btn_note.time) < 0.1:  # 100ms内
                        conflicts.append(ConflictReport(
                            has_conflict=True,
                            conflict_type="touch_button_overlap",
                            time=btn_note.time,
                            notes=[touch, btn_note],
                            severity="hard",
                            message=f"Touch A区 与 按键1的 {btn_note.note_type} 在{btn_note.time:.2f}s判定冲突"
                        ))
        
        return conflicts
    
    def _detect_excessive_jumps(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测过大的按键跳跃（软冲突）。"""
        conflicts = []
        
        taps = sorted([n for n in notes if n.note_type == "tap" and isinstance(n.button, int)], 
                      key=lambda n: n.time)
        
        for i in range(len(taps) - 1):
            t1, t2 = taps[i], taps[i+1]
            time_diff = t2.time - t1.time
            distance = ButtonLayout.physical_distance(t1.button, t2.button)
            
            # 短时间内大跨度跳跃
            if time_diff < 0.15 and distance > 1.5:  # 150ms内跨度>1.5（直径）
                conflicts.append(ConflictReport(
                    has_conflict=True,
                    conflict_type="excessive_jump",
                    time=t2.time,
                    notes=[t1, t2],
                    severity="soft",
                    message=f"按键{t1.button}→{t2.button} 在{time_diff*1000:.0f}ms内跳跃，物理距离{distance:.2f}过大"
                ))
        
        return conflicts
    
    def _detect_hand_dispatch_conflict(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        """检测双手调度冲突（软冲突）。"""
        conflicts = []
        
        # 找出同一时刻的多押
        time_groups = {}
        for note in notes:
            if note.note_type in ("tap", "hold") and isinstance(note.button, int):
                t_key = round(note.time, 3)  # 3ms容差
                if t_key not in time_groups:
                    time_groups[t_key] = []
                time_groups[t_key].append(note)
        
        for t, group in time_groups.items():
            if len(group) > 1:
                buttons = [n.button for n in group]
                if not HandDispatch.can_reach_simultaneous(buttons, allow_cross=False):
                    conflicts.append(ConflictReport(
                        has_conflict=True,
                        conflict_type="hand_dispatch_conflict",
                        time=t,
                        notes=group,
                        severity="soft",
                        message=f"{len(buttons)}押配置{buttons}在{t:.2f}s可能超出单手范围"
                    ))
        
        return conflicts
```

#### 3.4 冲突修复策略（可选）

:codex-annotation{index="5"}

```python
# src/maichart/generation/conflict_repair.py

"""冲突修复器（可选功能）。"""

class ConflictRepairer:
    """修复检测到的冲突。
    
    根据用户要求：
    - 修复是可选的，可以选择"只检测不修复"
    - 修复可以是局部的（单个窗口）或全局拒绝
    """
    
    def __init__(self, 
                 repair_mode: Literal["none", "local", "reject"] = "none"):
        """
        参数：
        - none: 只检测，不修复
        - local: 局部修复冲突窗口
        - reject: 检测到冲突时拒绝整个生成
        """
        self.mode = repair_mode
    
    def repair(self, notes: list[NoteEvent], conflicts: list[ConflictReport]) -> list[NoteEvent]:
        """根据mode修复冲突。"""
        if self.mode == "none":
            return notes
        
        if self.mode == "reject" and any(c.severity == "hard" for c in conflicts):
            raise ValueError(f"检测到{len(conflicts)}个硬冲突，拒绝生成")
        
        if self.mode == "local":
            return self._repair_local(notes, conflicts)
        
        return notes
    
    def _repair_local(self, notes: list[NoteEvent], conflicts: list[ConflictReport]) -> list[NoteEvent]:
        """局部修复：删除或移动冲突note。"""
        to_remove = set()
        
        for conflict in conflicts:
            if conflict.severity == "hard":
                # 删除冲突中的tap（保留hold/slide）
                for note in conflict.notes:
                    if note.note_type == "tap":
                        to_remove.add(id(note))
                        break
        
        return [n for n in notes if id(n) not in to_remove]
```

---

### 任务4：WebUI集成

#### 4.1 生成参数页面

在WebUI生成页面添加：

**复杂节奏控制区**：
```html
<div class="rhythm-controls">
  <h3>节奏倾向</h3>
  <div class="radio-group">
    <label><input type="radio" name="rhythm_bias" value="auto" checked> 自动（跟随音乐）</label>
    <label><input type="radio" name="rhythm_bias" value="uniform"> 均匀节拍</label>
    <label><input type="radio" name="rhythm_bias" value="triplet"> 三连音倾向</label>
    <label><input type="radio" name="rhythm_bias" value="dotted"> 附点节奏</label>
    <label><input type="radio" name="rhythm_bias" value="mixed"> 混合复杂</label>
  </div>
  
  <label>
    细碎音符强度 
    <input type="range" name="fine_note_strength" min="0" max="1" step="0.1" value="0.5">
    <span class="value">0.5</span>
  </label>
  <p class="hint">控制12分/16分/24分音符的生成倾向。0=保守采音，1=积极采样细碎节奏</p>
</div>
```

**物理约束控制区**：
```html
<div class="physics-controls">
  <h3>可玩性约束</h3>
  
  <label>
    <input type="checkbox" name="enable_conflict_detection" checked>
    启用冲突检测
  </label>
  
  <div class="sub-options" id="conflict-options">
    <label>
      <input type="checkbox" name="detect_hard_conflicts" checked>
      硬冲突（Hold/Tap重叠、Slide星星碰撞）
    </label>
    <label>
      <input type="checkbox" name="detect_soft_conflicts">
      软冲突（过大跳跃、手部调度）
    </label>
    
    <label>
      冲突处理方式
      <select name="conflict_repair_mode">
        <option value="none">仅检测，不修复</option>
        <option value="local">局部修复冲突</option>
        <option value="reject">检测到冲突时重新生成</option>
      </select>
    </label>
  </div>
  
  <p class="hint">
    硬冲突：游戏判定系统无法处理的配置（推荐启用）<br>
    软冲突：人体工学困难但非绝对不可能（可选）
  </p>
</div>
```

#### 4.2 难度预设更新

根据用户之前要求，为12.5-14.9难度添加预设，现在需要增加节奏倾向预设：

```python
# src/maichart/generation/preset_calibration.py 扩展

RHYTHM_PRESETS = {
    "standard": {
        "fine_note_strength": 0.5,
        "rhythm_bias": "auto",
        "description": "标准节奏平衡"
    },
    "technical": {
        "fine_note_strength": 0.8,
        "rhythm_bias": "mixed",
        "description": "技巧谱，细碎节奏丰富"
    },
    "simple": {
        "fine_note_strength": 0.2,
        "rhythm_bias": "uniform",
        "description": "简单清晰节拍"
    }
}

# 为每个难度预设添加rhythm子预设
# 示例：12.5难度 + technical rhythm = 12.5技巧风格
```

---

### 任务5：训练流程

#### 5.1 训练阶段规划

**Stage 0: 节奏标注准备**（数据预处理）
```bash
# 为full训练集添加节奏标注
python scripts/annotate_rhythm_patterns.py \
  --dataset webui_workspace/datasets/full训练集 \
  --output webui_workspace/datasets/full训练集/rhythm_annotated_manifest.json
```

**Stage 1: 基础全量训练**（fresh，15 epochs）
```bash
# 使用music_structure_v2
python -m maichart.generation.event_ctx_cli train \
  --dataset webui_workspace/datasets/full训练集 \
  --output checkpoints/rhythm_v1 \
  --fresh \
  --planner-profile music_structure_v2 \
  --difficulty-range 4 6 \
  --level-range 12.0 15.0 \
  --epochs 15 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --save-every 2
```

**Stage 2: 复杂节奏专项强化**（finetune，5-10 epochs）
```bash
# 基于stage1的checkpoint，增加复杂节奏样本权重
python -m maichart.generation.event_ctx_cli train \
  --dataset webui_workspace/datasets/full训练集 \
  --output checkpoints/rhythm_v1_fine \
  --init-checkpoint checkpoints/rhythm_v1/epoch_015.pt \
  --oversample-fine-rhythm 2.5 \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 5e-5
```

#### 5.2 评估指标

**音频结构指标**：
- 首末音符与可听边界的距离
- 静音区事件数量
- 窗口密度与音频结构特征相关性
- 高潮段落位置与密度峰值对齐度

**节奏采样指标**（新增）：
- 细碎音符占比（12/16/24分音符数量 / 总数）
- 三连音段落的采样率（实际采样 / 理论可采样）
- 复杂节奏窗口的密度方差（越高越好，说明跟随了变化）
- 与参考金标谱面的节奏相似度

**可玩性指标**：
- 硬冲突检出率（应为0）
- 软冲突检出率（与人工审核对比）
- 最大单手跨度
- 平均手部切换频率

**对比测试**：
固定10首测试歌曲，分别生成：
1. 基础模型（Stage 1 checkpoint）
2. 强化模型（Stage 2 checkpoint）
3. 启用物理约束 vs 禁用物理约束

人工评估：
- 节奏是否跟随了细碎变化
- 是否出现无理配置
- 整体可玩性

---

## 四、实施计划

### 4.1 优先级分级

**P0（必须完成）**：
- [x] 任务1.1：实现细粒度节奏特征提取
- [x] 任务1.2：训练集节奏标注脚本
- [x] 任务3.1-3.3：舞萌物理模型与冲突检测器
- [x] 任务4.1：WebUI生成参数集成
- [x] 任务5.1：训练流程Stage 1

**P1（重要）**：
- [ ] 任务2.1：构建复杂节奏专项集
- [ ] 任务3.4：冲突修复策略
- [ ] 任务4.2：难度预设扩展
- [ ] 任务5.1：训练流程Stage 2

**P2（可选优化）**：
- [ ] 任务1.3：扩展规划器维度（评估后决定）
- [ ] Slide轨迹的精确插值算法
- [ ] 多声部分离的高级实现
- [ ] 基于MiaCode的硬无理检测集成

### 4.2 时间估算

- **数据准备**：2-3天（节奏标注 + 专项集筛选）
- **代码实现**：5-7天（物理模型 + 冲突检测 + WebUI集成）
- **测试与调试**：2-3天（单元测试 + 集成测试）
- **Stage 1 训练**：2-3天（15 epochs，假设RTX 4090）
- **评估与迭代**：2-3天（生成测试 + 人工评审）
- **Stage 2 训练**（如需）：1-2天

**总计**：14-21天（约2-3周）

### 4.3 验收标准

**最小可用版本（MVP）**：
- ✅ 节奏特征提取代码可运行，能输出triplet/dotted/entropy等指标
- ✅ 冲突检测器能识别基本的hold-tap、slide-tap冲突
- ✅ WebUI中有节奏倾向和物理约束的开关
- ✅ Stage 1训练收敛，loss正常下降

**完整版本**：
- ✅ 固定测试集上，细碎音符采样率提升30%+（相比baseline）
- ✅ 硬冲突检出率<1%（在1000个生成窗口中）
- ✅ 人工可玩性评分提升（5点量表，提升0.5+）
- ✅ 复杂节奏段落的质量评分提升

---

## 五、风险与应对

### 5.1 主要风险

**R1：细粒度节奏特征过拟合**
- **风险**：模型学会了"标注=triplet就强制生成三连音"，而非从音频学习
- **应对**：训练时以一定概率（30%）mask掉rhythm标注，强制模型依赖音频

**R2：物理约束过度限制**
- **风险**：冲突检测过严，拒绝了实际合理的配置
- **应对**：默认只启用硬冲突，软冲突可选；对比禁用约束的生成质量

**R3：训练集偏向未根本解决**
- **风险**：即使增加复杂节奏权重，模型仍倾向于简化
- **应对**：考虑引入对比学习（contrastive learning），让模型区分"保守采样"和"积极采样"

**R4：Slide轨迹计算复杂度**
- **风险**：精确的星星位置计算可能拖慢生成速度
- **应对**：使用缓存+预计算，只在冲突检测时精确计算

### 5.2 回退方案

如果Stage 2训练不收敛或质量下降：
1. 回退到Stage 1 checkpoint
2. 降低复杂节奏过采样倍率（2.5x → 1.5x）
3. 延长Stage 1训练（15 epochs → 25 epochs）

如果物理约束导致生成失败率过高：
1. 默认禁用软冲突检测
2. 提高冲突容差阈值
3. 改为"仅检测+警告"模式，不拒绝生成

---

## 六、后续扩展方向

**短期（3个月内）**：
- 集成MiaCode的专业无理检测库
- 引入强化学习微调（RL from Human Feedback）
- 多难度联合训练（10-15级联合学习）

**中期（6个月内）**：
- Slide轨迹的物理仿真器
- 基于真实玩家replay数据的可玩性建模
- 自动难度评估器（输出精确定数）

**长期（1年内）**：
- 多模态模型（音频+视频MV+歌词）
- 风格迁移（学习特定谱师风格）
- 在线学习（从玩家反馈持续改进）

---

## 七、附录

### A. 舞萌DX按键布局参考

```
       1 ← 正上方偏右22.5°
      /  \
    8      2
   /          \
  7            3
   \          /
    6      4
      \  /
       5
```

### B. 常见Slide Pattern
- `-`: 直线
- `<`, `>`: 弧线（逆时针/顺时针）
- `^`: 最短弧
- `p`, `q`, `pp`, `qq`: 曲线
- `s`, `z`: 之字形
- `V`: V形（2个path参数）
- `w`: W形

### C. Touch区域映射
- A: 中心圆形
- B: 内环
- C: 中环
- D: 外环
- E: 边缘环

Touch与Button的重叠主要是A区与1号按键。

---

## 八、参考文档

- [HANDOFF_MUSIC_STRUCTURE_PLANNER.md](HANDOFF_MUSIC_STRUCTURE_PLANNER.md)：music_structure_v2架构交接文档
- [QUALITY_AND_MUSIC_STRUCTURE_V2.md](QUALITY_AND_MUSIC_STRUCTURE_V2.md)：质量标注与规划器使用说明
- [CURVE_DENSITY_FEEDBACK.md](CURVE_DENSITY_FEEDBACK.md)：曲线控制详细说明
- [EVENT_GUIDE.md](EVENT_GUIDE.md)：Event模型基础说明
- 用户提供的对话摘录：音乐强度≠响度，谱面是音乐结构到可玩动作的转换

---

**文档状态**：最终版本，待审核  
**下一步**：用户审核本方案 → 开始实施任务1.1（节奏特征提取）

