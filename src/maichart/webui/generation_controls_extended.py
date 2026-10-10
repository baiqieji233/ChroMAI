"""WebUI生成参数配置扩展 - 节奏倾向和可玩性约束。

这个配置应该添加到WebUI前端的生成参数面板中。
"""

# 节奏倾向控制配置
RHYTHM_CONTROLS = {
    "name": "rhythm_bias",
    "label": "节奏倾向",
    "description": "控制模型对复杂节奏的采样倾向",
    "type": "select",
    "options": [
        {
            "value": "auto",
            "label": "自动（跟随音乐）",
            "description": "模型根据音频特征自动判断节奏倾向"
        },
        {
            "value": "uniform",
            "label": "均匀节拍",
            "description": "优先采样4分/8分等均匀间隔的音符"
        },
        {
            "value": "triplet",
            "label": "三连音倾向",
            "description": "增加三连音（1/3拍）节奏的采样概率"
        },
        {
            "value": "dotted",
            "label": "附点节奏",
            "description": "增加附点节奏（3:1时长比）的采样概率"
        },
        {
            "value": "mixed",
            "label": "混合复杂",
            "description": "鼓励混合使用多种节奏模式"
        }
    ],
    "default": "auto"
}

FINE_NOTE_STRENGTH = {
    "name": "fine_note_strength",
    "label": "细碎音符强度",
    "description": "控制12分/16分/24分等细碎音符的生成倾向。0=保守采音，1=积极采样",
    "type": "slider",
    "min": 0.0,
    "max": 1.0,
    "step": 0.1,
    "default": 0.5
}

# 可玩性约束配置
CONFLICT_DETECTION = {
    "name": "enable_conflict_detection",
    "label": "启用冲突检测",
    "description": "在生成后检测可玩性问题（硬无理、伪无理等）",
    "type": "checkbox",
    "default": True
}

DETECT_HARD_CONFLICTS = {
    "name": "detect_hard_conflicts",
    "label": "检测硬冲突",
    "description": "Hold-Tap重叠、Slide-Tap碰撞、Touch-Button判定冲突等游戏判定系统无法处理的配置",
    "type": "checkbox",
    "default": True,
    "depends_on": "enable_conflict_detection"
}

DETECT_SOFT_CONFLICTS = {
    "name": "detect_soft_conflicts",
    "label": "检测软冲突",
    "description": "过大跳跃、手部调度困难等人体工学问题（可选）",
    "type": "checkbox",
    "default": False,
    "depends_on": "enable_conflict_detection"
}

CONFLICT_REPAIR_MODE = {
    "name": "conflict_repair_mode",
    "label": "冲突处理方式",
    "description": "检测到冲突时的处理方式",
    "type": "select",
    "options": [
        {
            "value": "none",
            "label": "仅检测，不修复",
            "description": "报告冲突但保留生成的谱面（用于分析）"
        },
        {
            "value": "local",
            "label": "局部修复",
            "description": "删除冲突的note，保留其他内容"
        },
        {
            "value": "reject",
            "label": "检测到硬冲突时拒绝",
            "description": "发现硬冲突则放弃该窗口的生成，要求重新生成"
        }
    ],
    "default": "none",
    "depends_on": "enable_conflict_detection"
}

# 完整的WebUI扩展配置
EXTENDED_GENERATION_CONTROLS = {
    "sections": [
        {
            "title": "节奏采样控制",
            "description": "控制模型对复杂节奏的学习和生成",
            "controls": [
                RHYTHM_CONTROLS,
                FINE_NOTE_STRENGTH
            ]
        },
        {
            "title": "可玩性约束",
            "description": "防止生成无理或伪无理配置",
            "controls": [
                CONFLICT_DETECTION,
                DETECT_HARD_CONFLICTS,
                DETECT_SOFT_CONFLICTS,
                CONFLICT_REPAIR_MODE
            ]
        }
    ]
}

# 前端HTML代码示例
FRONTEND_HTML_RHYTHM_SECTION = """
<div class="control-section rhythm-controls">
  <h3>节奏采样控制</h3>
  <p class="section-help">控制模型对复杂节奏的学习和生成。目前模型在面对三连音、附点、多声部等复杂节奏时倾向保守采音。</p>
  
  <div class="control-group">
    <label>节奏倾向</label>
    <select id="rhythm_bias" name="rhythm_bias" onchange="updateGenerationConfig()">
      <option value="auto" selected>自动（跟随音乐）</option>
      <option value="uniform">均匀节拍</option>
      <option value="triplet">三连音倾向</option>
      <option value="dotted">附点节奏</option>
      <option value="mixed">混合复杂</option>
    </select>
    <p class="control-help">自动模式：模型根据音频特征判断，通常最稳定</p>
  </div>
  
  <div class="control-group">
    <label>细碎音符强度<span id="fine_note_value">0.5</span></label>
    <input type="range" id="fine_note_strength" name="fine_note_strength" 
           min="0" max="1" step="0.1" value="0.5" 
           oninput="updateFineNoteValue()" onchange="updateGenerationConfig()">
    <p class="control-help">
      <strong>0</strong> = 保守采音，只用4分8分<br>
      <strong>0.5</strong> = 平衡，偶尔采样细碎节奏<br>
      <strong>1.0</strong> = 积极采样，12分16分24分等都可能出现
    </p>
  </div>
</div>
"""

FRONTEND_HTML_CONFLICT_SECTION = """
<div class="control-section conflict-controls">
  <h3>可玩性约束</h3>
  <p class="section-help">防止生成无理配置。"硬无理"是游戏判定系统无法处理的配置，"软冲突"是人体工学困难但非绝对不可能。</p>
  
  <div class="control-group">
    <label>
      <input type="checkbox" id="enable_conflict_detection" name="enable_conflict_detection" 
             checked onchange="updateConflictControls()">
      启用冲突检测
    </label>
    <p class="control-help">检测生成谱面中的无理配置</p>
  </div>
  
  <div class="sub-options" id="conflict-options">
    <div class="control-group">
      <label>
        <input type="checkbox" id="detect_hard_conflicts" name="detect_hard_conflicts" 
               checked onchange="updateGenerationConfig()">
        检测硬冲突（Hold-Tap重叠、Slide星星碰撞等）
      </label>
      <p class="control-help">推荐始终启用，这些配置在游戏中无法正确判定</p>
    </div>
    
    <div class="control-group">
      <label>
        <input type="checkbox" id="detect_soft_conflicts" name="detect_soft_conflicts" 
               onchange="updateGenerationConfig()">
        检测软冲突（过大跳跃、手部调度困难）
      </label>
      <p class="control-help">可选，基于人体工学的检测，有时可能过于严格</p>
    </div>
    
    <div class="control-group">
      <label>冲突处理方式</label>
      <select id="conflict_repair_mode" name="conflict_repair_mode" onchange="updateGenerationConfig()">
        <option value="none" selected>仅检测，不修复</option>
        <option value="local">局部修复（删除冲突note）</option>
        <option value="reject">检测到硬冲突时拒绝</option>
      </select>
      <p class="control-help">
        <strong>仅检测</strong>：报告冲突位置，保留谱面<br>
        <strong>局部修复</strong>：自动删除冲突的note<br>
        <strong>拒绝</strong>：发现硬冲突则要求重新生成
      </p>
    </div>
  </div>
</div>
"""

# JavaScript对应的配置处理函数
FRONTEND_JS_CONFIG_HANDLER = """
function updateFineNoteValue() {
  const value = document.getElementById("fine_note_strength").value;
  document.getElementById("fine_note_value").textContent = value;
}

function updateConflictControls() {
  const enabled = document.getElementById("enable_conflict_detection").checked;
  const options = document.getElementById("conflict-options");
  const subControls = options.querySelectorAll("input, select");
  
  subControls.forEach(ctrl => {
    ctrl.disabled = !enabled;
  });
}

function getGenerationConfig() {
  const config = {
    // 原有配置...
    
    // 新增节奏采样配置
    rhythm_bias: document.getElementById("rhythm_bias")?.value || "auto",
    fine_note_strength: parseFloat(document.getElementById("fine_note_strength")?.value || "0.5"),
    
    // 新增可玩性约束配置
    enable_conflict_detection: document.getElementById("enable_conflict_detection")?.checked || true,
    detect_hard_conflicts: document.getElementById("detect_hard_conflicts")?.checked || true,
    detect_soft_conflicts: document.getElementById("detect_soft_conflicts")?.checked || false,
    conflict_repair_mode: document.getElementById("conflict_repair_mode")?.value || "none",
  };
  
  return config;
}

function updateGenerationConfig() {
  const config = getGenerationConfig();
  // 发送到后端或保存到localStorage
  console.log("Generation config updated:", config);
  // localStorage.setItem("generation_config", JSON.stringify(config));
}
"""

if __name__ == "__main__":
    import json
    
    print("=== WebUI配置示例 ===\n")
    print("节奏倾向配置:")
    print(json.dumps(RHYTHM_CONTROLS, indent=2, ensure_ascii=False))
    
    print("\n可玩性约束配置:")
    print(json.dumps(CONFLICT_DETECTION, indent=2, ensure_ascii=False))
    
    print("\n完整配置结构:")
    print(json.dumps(EXTENDED_GENERATION_CONTROLS, indent=2, ensure_ascii=False))
