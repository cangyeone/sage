<p align="center">
  <img src="logo.png" alt="SeismicXM logo"/>
</p>

---



# SAGE — Seismology AI-Guided Engine

> 面向地震学研究的对话式 AI 分析平台

SAGE 是一个集**自然语言交互**、**智能震相拾取**、**统计分析**、**代码生成执行**和**文献解读**于一体的地震学 AI 平台。用户可以通过中文对话驱动整个分析流程，无需记忆命令行参数或编写样板代码。

---

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [安装](#安装)
- [配置 LLM 后端](#配置-llm-后端)
- [Web 界面](#web-界面)
- [命令行工具](#命令行工具)
- [seismo\_skill 技能系统](#seismo_skill-技能系统)
- [核心模块](#核心模块)
- [目录结构](#目录结构)
- [配置文件](#配置文件)
- [pnsn 震相拾取模块](#pnsn-震相拾取模块)
- [常见问题](#常见问题)

---

## 功能概览

| 模块 | 功能 |
|------|------|
| 💬 **对话 Agent** | 自然语言理解 → 意图分类（LLM 优先 + 规则兜底）→ 执行对应技能 |
| 🔍 **震相拾取** | 单台在线拾取 / 目录批量拾取，支持 JIT 与 ONNX 多种模型 |
| 🔗 **震相关联** | FastLink / REAL / Gamma 多种方法，将台站拾取结果关联为地震事件 |
| 🧭 **极性分析** | P 波初动极性自动判断 |
| 📊 **地震统计** | b 值估算（MLE/LSQ）、F-M 分布图、时序分布、空间分布 |
| 🧑‍💻 **代码生成** | LLM 生成 Python 代码 + 沙箱安全执行，内置地震学工具包 |
| 🤖 **自主 Agent** | 读入论文 → 理解方法 → 自主规划 → 逐步编程实现 |
| 📚 **知识库 RAG** | BGE-M3 向量化 + FAISS 检索，支持批量 PDF 入库和文献问答 |
| 📖 **文献解读** | 临时上传 PDF → RAG 对话 / 深度解读 |
| 🗂 **本地文件访问** | 授权指定目录后，LLM 可直接读取文件列表辅助分析 |
| ⚡ **技能系统** | Markdown 格式的函数使用说明文档，AI 对话和代码生成时自动检索并参考 |
| 📈 **波形可视化** | 对话窗口内嵌 Plotly 交互式波形图，震相标注叠加显示 |

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│                      Web UI (Flask)                            │
│   /chat  ·  /knowledge  ·  /skills  ·  /llm-settings          │
└────────────────────────┬───────────────────────────────────────┘
                         │ HTTP API
┌────────────────────────▼───────────────────────────────────────┐
│                 ConversationalAgent                             │
│  ┌──────────────────┐   ┌────────────────────────────────────┐ │
│  │ IntentClassifier │   │         SkillExecutor              │ │
│  │  LLM 优先        │──▶│  batch_picking / seismo_qa         │ │
│  │  规则兜底        │   │  seismo_programming / agent        │ │
│  └──────────────────┘   └────────────────────────────────────┘ │
└──────────┬──────────────────────────┬──────────────────────────┘
           │                          │
┌──────────▼──────────┐   ┌──────────▼─────────────────────────┐
│  pnsn/ 震相拾取     │   │  seismo_code / seismo_stats        │
│  ONNX / JIT 模型    │   │  seismo_agent / seismo_tools       │
└─────────────────────┘   └──────────┬─────────────────────────┘
                                     │ 自动检索技能文档
                          ┌──────────▼─────────────────────────┐
                          │       seismo_skill/                 │
                          │  内置技能文档  +  用户自定义技能     │
                          │  (~/.seismicx/skills/)              │
                          └────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────┐
│                LLM Backend (config_manager)                     │
│      Ollama (本地)  ·  vLLM  ·  OpenAI / DeepSeek             │
└────────────────────────────────────────────────────────────────┘
```

---

## 安装

### 系统要求

- Python ≥ 3.9
- 内存 ≥ 8 GB（运行本地 LLM 时建议 16 GB+）
- GPU 可选（震相拾取和 LLM 均支持 CPU 模式）

### 基础安装

```bash
git clone https://github.com/yourname/sage.git
cd sage

# 安装完整依赖（推荐）
pip install -r requirements.txt

# 或者手动安装各部分依赖
# Web 界面依赖
pip install flask flask-cors

# 震相拾取相关
pip install obspy torch onnxruntime tqdm matplotlib

# RAG 知识库依赖（可选，首次使用时自动下载 BGE-M3 模型约 2 GB）
pip install FlagEmbedding faiss-cpu sentence-transformers pdfminer.six PyMuPDF
```

### pnsn 震相拾取模块安装

SAGE 集成了先进的震相拾取模块，需要单独克隆并配置：

```bash
# 克隆 pnsn 震相拾取模块
git clone https://github.com/cangyeone/pnsn.git

# 安装 pnsn 依赖
cd pnsn
pip install -r requirements.txt

# 返回主项目目录
cd ../
```

pnsn 模块包含多种震相拾取模型：
- **PhaseNet**：用于 P/S 波初至拾取
- **EQTransformer**：用于地震事件检测和震相拾取
- **JMA Picker**：日本气象厅震相拾取模型

### 重要：RAG 功能依赖（嵌入模型库）

如果需要使用知识库 RAG 功能（PDF 文档问答、文献解读等），需要额外安装 Rust 编译器以构建必要的库：

```bash
# 安装 Rust（用于构建 tokenizers 库）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 然后安装嵌入模型相关依赖
pip install transformers==4.33.0 FlagEmbedding sentence-transformers
```

### 启动 Web 服务

```bash
python web_app/app.py --port 5010
# 浏览器访问 http://localhost:5010
```

---

## 配置 LLM 后端

SAGE 所有 AI 功能（意图分类、问答、代码生成、文献解读）均需要 LLM 后端。配置通过 **Web 界面 → LLM 设置页** 或命令行完成，统一保存在 `~/.seismicx/config.json`，所有功能实时共享同一份配置。

### 方式一：Ollama（推荐，本地运行，无需联网）

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动服务
ollama serve

# 拉取模型（任选一个）
ollama pull qwen3:30b        # 综合能力强，需 ~20 GB 内存
ollama pull qwen3:8b         # 轻量，需 ~6 GB 内存
ollama pull deepseek-r1:8b   # 推理能力强
ollama pull llama3.3:latest  # 英文能力强
```

然后在 **LLM 设置页** 中选择对应模型并点击「保存配置」即可。

### 方式二：在线 API（OpenAI 兼容格式）

在 **LLM 设置页** 选择「自定义 API」并填写：

| 参数 | 示例值 |
|------|--------|
| API Base URL | `https://api.deepseek.com/v1` |
| API Key | `sk-xxxxxxxx` |
| 模型名称 | `deepseek-chat` |

支持任意 OpenAI 兼容接口，包括 DeepSeek、SiliconFlow、月之暗面（Moonshot）、阿里通义（DashScope）等。

### 方式三：命令行配置

```bash
# 使用 Ollama 本地模型
python seismic_cli.py backend use ollama --model qwen3:30b

# 使用在线 API
python seismic_cli.py backend use online \
    --provider deepseek --api-key sk-xxx --model deepseek-chat

# 查看所有后端状态
python seismic_cli.py backend status

# 自动检测并选择可用后端
python seismic_cli.py backend auto
```

---

## Web 界面

启动后访问 `http://localhost:5010`，包含四个页面：

### 对话页（/chat）

主交互界面，支持三种模式：

- **💬 对话模式**：自然语言问答，支持震相拾取、数据处理、地震学知识等全部技能
- **📖 文献解读模式**：上传 PDF 后逐段解读，提取方法、公式、结论
- **🔍 知识检索模式**：从持久知识库中向量检索后精准回答

**侧边栏功能：**
- 📎 上传 PDF 供当前会话临时使用
- 🗂 授权本地工作目录（LLM 可读取指定路径下的文件列表）
- 知识库状态显示与快速入口

**技能自动注入：** 对话时，系统会自动检索与当前问题最相关的技能文档（来自 `seismo_skill/`），将函数说明和示例代码注入 LLM 上下文，让回答更精准、代码更规范。

**对话示例：**

```
> 如何用 ObsPy 下载 IRIS 波形数据？
> 帮我对 /data/wave.mseed 做带通滤波 1-10 Hz 并画图
> 批量拾取 /data/seismic/2024/ 下的所有波形文件
> 计算 catalog.csv 中的 b 值并绘制 F-M 分布图
> 查看下目录 /data/seismic/waveform 中的文件
> 阅读上传的论文，实现其中的 HVSR 谱比法
```

### 知识库页（/knowledge）

- 拖拽上传多个 PDF，自动使用 **BGE-M3** 向量化后入库
- 实时显示索引进度（文本提取 → 分块 → 嵌入 → FAISS 写入）
- 文献管理：查看页数 / 片段数 / 文件大小，支持删除、清空

> 知识库持久存储于 `~/.seismicx/knowledge/`，重启服务后仍可用。

### 技能管理页（/skills）

查看和管理所有技能文档，无需重启服务即可扩展 AI 能力。

**功能：**
- 左侧列表：内置技能（灰色标记）与自定义技能（绿色标记）分组展示
- 右侧编辑器：支持 Markdown 编辑和实时预览（含语法高亮）
- 内置技能只读；自定义技能可随时编辑或删除
- 「新建」弹窗：填写技能 ID、标题、关键词和描述，自动生成模板并进入编辑模式

**技能文件格式（Markdown + YAML 前置元数据）：**

```
---
name: my_custom_tool
category: custom
keywords: 关键词1, 关键词2, english_keyword
---

# 我的自定义工具

## 描述

工具功能说明。

## 主要函数

### `my_function(param1, param2)`

**参数：**
- `param1` : type — 说明

```python
# 示例代码
result = my_function(param1, param2)
print(result)
```
```

> 技能文件保存于 `~/.seismicx/skills/`，在所有 SAGE 实例间共享。

### LLM 设置页（/llm-settings）

- 在线检测 Ollama 已安装模型并一键选择
- 支持配置在线 API（自定义 Base URL + Key + 模型名）
- 保存后立即对所有功能生效，**无需重启服务**
- 顶部徽章实时显示当前生效的模型

---

## 命令行工具

`seismic_cli.py` 提供完整的命令行接口，适合脚本化和批量处理场景。

### 对话模式

```bash
python seismic_cli.py chat
```

### 震相拾取

```bash
# 单台拾取
python seismic_cli.py pick -i /data/station/ -m pnsn/pickers/pnsn.v3.jit

# 批量拾取（整个目录）
python seismic_cli.py pick -i /data/seismic/2024/ --batch -o results/picks.csv

# 指定设备
python seismic_cli.py pick -i /data/ --device cuda
```

### 震相关联

```bash
python seismic_cli.py associate \
    -i results/picks.csv \
    -s station_list.csv \
    --method fastlink \
    -o results/events.txt
```

### 地震统计

```bash
# 计算 b 值
python seismic_cli.py stats bvalue -i catalog.csv --mc auto

# 绘制 F-M 分布图
python seismic_cli.py stats plot-gr -i catalog.csv -o fmd.png

# 生成完整统计报告（b 值 + 时序 + 空间分布）
python seismic_cli.py stats report -i catalog.csv
```

### LLM 代码生成执行

```bash
python seismic_cli.py run "对 /data/wave.mseed 做 1-10Hz 带通滤波并画图"
python seismic_cli.py run "计算震源参数，震中距 50km" -d /data/waves/
python seismic_cli.py run "画走时曲线，距离 0-30°，深度 10km" --show-code
```

### 自主 Agent

```bash
# 从本地 PDF 实现算法
python seismic_cli.py agent \
    "实现论文中的走时残差校正方法" \
    --paper /papers/velest_method.pdf \
    --data /data/picks.csv \
    --output results/agent_run/

# 从 arXiv 论文 ID 实现
python seismic_cli.py agent \
    "复现论文的 b 值时序分析方法" \
    --arxiv 2309.12345

# 从 DOI 实现
python seismic_cli.py agent \
    "实现 HVSR 谱比法" \
    --doi 10.1785/0220230045 \
    --max-steps 6
```

### 技能管理

```bash
# 列出所有技能（内置 + 自定义，分组显示）
python seismic_cli.py skill list

# 按关键词搜索技能（支持中英文混合）
python seismic_cli.py skill search "带通滤波"
python seismic_cli.py skill search "b-value Gutenberg"

# 查看技能完整文档
python seismic_cli.py skill show waveform_processing

# 新建用户自定义技能（生成模板后打开 $EDITOR）
python seismic_cli.py skill new my_tool \
    --title "我的工具" \
    --keywords "关键词1, keyword2" \
    --desc "功能描述"

# 编辑已有自定义技能
python seismic_cli.py skill edit my_tool

# 删除自定义技能
python seismic_cli.py skill delete my_tool

# 查看用户技能目录路径
python seismic_cli.py skill dir
```

### LLM 后端管理

```bash
python seismic_cli.py backend status           # 查看当前状态
python seismic_cli.py backend setup            # 交互式配置向导
python seismic_cli.py backend auto             # 自动检测并选择
python seismic_cli.py backend models           # 列出本地已下载模型
python seismic_cli.py backend pull qwen3:8b    # 拉取 Ollama 模型
```

---

## seismo_skill 技能系统

技能系统是 SAGE 的核心扩展机制。每个技能是一个 Markdown 文档，描述某类函数或工具的使用方式和示例代码。**AI 在生成代码或回答问题时会自动检索并参考最相关的技能文档**，无需用户手动指定。

### 工作原理

```
用户输入（自然语言）
       │
       ▼
  seismo_skill 检索
  （中英文混合关键词匹配）
       │
       ├─ 匹配到技能文档 → 注入系统提示（函数签名 + 示例代码）
       │
       ▼
  LLM 生成代码 / 回答
  （优先使用技能文档中的规范写法）
```

三个触发点均已集成技能检索：
- `/api/chat/rag`（Web 对话）
- `seismo_code/code_engine.py`（`run` 命令代码生成）
- `seismo_agent/agent_loop.py`（Agent 每步代码生成）

### 内置技能（6 个）

| 技能文件 | 类别 | 主要函数 |
|----------|------|----------|
| `waveform_io.md` | waveform | `read_stream`, `read_stream_from_dir`, `stream_info`, `picks_to_dict` |
| `waveform_processing.md` | waveform | `detrend_stream`, `taper_stream`, `filter_stream`, `resample_stream`, `trim_stream`, `remove_response` |
| `waveform_visualization.md` | visualization | `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion` |
| `spectral_analysis.md` | analysis | `compute_spectrum`, `compute_hvsr` |
| `b_value_analysis.md` | statistics | `load_catalog_file`, `calc_mc_*`, `calc_bvalue_mle`, `plot_gr` |
| `source_parameters.md` | analysis | `estimate_magnitude_ml`, `estimate_corner_freq`, `estimate_seismic_moment`, `moment_to_mw`, `estimate_stress_drop` |

### 创建自定义技能

**方式一：Web 界面**（推荐）

访问 `/skills` → 点击「新建自定义技能」→ 填写基本信息 → 在编辑器中完善文档内容。

**方式二：命令行**

```bash
python seismic_cli.py skill new my_hypodd_tool \
    --title "HypoDD 双差定位工具" \
    --keywords "双差定位, HypoDD, 精定位, relocation" \
    --desc "封装 HypoDD 输入文件生成和结果解析"
# 自动打开 $EDITOR 编辑模板
```

**方式三：直接编写文件**

在 `~/.seismicx/skills/` 下创建任意 `.md` 文件，格式如下：

```
---
name: my_skill_name
category: custom
keywords: 关键词1, 关键词2, english_kw
---

# 技能标题

## 描述

功能说明（一两句话）。

---

## 主要函数

### `function_name(param1, param2=default)`

**参数：**
- `param1` : type — 说明
- `param2` : type — 说明，默认 default

**返回：** type — 说明

```
# 最小可运行示例
result = function_name("input", param2=42)
print(result)
```

---

## 注意事项

- 注意事项 1
- 注意事项 2
```

### 技能检索规则

- **关键词匹配**：支持中英文混合，自动对中文做单字 + 双字 bigram 分词
- **优先级**：用户自定义技能得分额外加权，相同匹配度下优先推荐
- **覆盖规则**：与内置技能同名的自定义技能会自动替换内置版本
- **实时生效**：新增 / 编辑技能文件后无需重启，下次请求即生效

---

## 核心模块

### `conversational_agent.py` — 对话核心

意图分类（15 类）→ 技能分发 → 结果返回。意图识别采用 **LLM 优先、规则兜底** 策略：LLM 在 10 秒内返回则使用 LLM 结果，超时或不可用时自动切换为关键词 + 正则规则分类。

支持的意图类型：

| 意图 | 触发示例 |
|------|---------|
| `seismo_qa` | "如何获取 IRIS 波形？" "b 值怎么计算？" |
| `seismo_programming` | "对数据做带通滤波" "去均值并重采样" |
| `seismo_agent` | "阅读这篇论文并实现其中的方法" |
| `batch_picking` | "批量拾取 /data/ 下的波形" |
| `phase_picking` | "帮我拾取震相" "检测 P 波和 S 波" |
| `waveform_plotting` | "画出波形图" "可视化台站数据" |
| `seismo_statistics` | "计算 b 值" "绘制时序分布图" |
| `data_browsing` | "查看 /data/ 下有哪些文件" |
| `configure` | "设置模型路径" "更改采样率" |

### `seismo_skill/` — 技能文档系统

按关键词为 LLM 提供函数使用说明和代码示例，提升生成代码的准确性和规范性。

```
seismo_skill/
├── skill_loader.py           # 核心：解析、检索、注入技能文档
├── __init__.py               # 公开接口
├── waveform_io.md            # 波形读取技能
├── waveform_processing.md    # 波形预处理技能
├── waveform_visualization.md # 波形可视化技能
├── spectral_analysis.md      # 频谱分析与 HVSR 技能
├── b_value_analysis.md       # b 值统计分析技能
└── source_parameters.md      # 震源参数估算技能

~/.seismicx/skills/           # 用户自定义技能目录（自动创建）
└── my_custom_skill.md
```

主要接口（`from seismo_skill import ...`）：

| 函数 | 说明 |
|------|------|
| `list_skills()` | 列出所有技能（含来源标记 builtin / user） |
| `search_skills(query, top_k)` | 中英文混合关键词检索，返回最相关技能列表 |
| `load_skill(name)` | 按名称加载完整 Markdown 文档 |
| `build_skill_context(query)` | 检索并格式化为可注入 LLM 提示的上下文块 |
| `save_user_skill(name, text)` | 保存用户自定义技能文件 |
| `delete_user_skill(name)` | 删除用户自定义技能（内置技能不可删除） |
| `get_skill_detail(name)` | 获取技能完整条目（含 source / path / body） |
| `invalidate_cache()` | 清除缓存，强制重新加载所有技能文件 |

### `seismo_agent/` — 自主 Agent

读入文献 → LLM 理解方法 → 自主规划步骤 → 逐步生成代码并执行 → 汇总报告。每个步骤生成代码前会自动检索相关技能文档，注入到生成提示中。

```
seismo_agent/
├── paper_reader.py   # 文献加载（PDF 本地 / arXiv ID / DOI / 纯文本）
├── memory.py         # 跨步骤工作记忆（已加载文献、步骤结果、变量）
├── planner.py        # LLM 任务规划（目标 + 文献 → JSON 步骤列表）
└── agent_loop.py     # 主循环（规划 → 代码生成 → 执行 → 失败重试）
```

执行流程：

```
用户目标 + 文献来源
       │
       ▼
  加载文献（PDF/arXiv/DOI）
       │
       ▼
  LLM 提取核心方法摘要
       │
       ▼
  规划执行步骤（3~8 步，JSON）
       │
   ┌───▼────────────────────────────────┐
   │  检索相关技能文档（seismo_skill）   │
   │  生成步骤代码（LLM + 技能上下文）  │  ← 最多重试 2 次
   │  沙箱安全执行                      │
   │  记录结果 / 图像                   │
   └───┬────────────────────────────────┘
       │ 循环所有步骤
       ▼
  汇总报告 + 输出目录
```

### `seismo_code/` — 代码生成与执行

- **`code_engine.py`**：LLM 代码生成，自动检索 seismo_skill 注入相关技能文档，并向 LLM 注入内置地震学工具包的 API 说明
- **`safe_executor.py`**：独立子进程沙箱执行，超时保护（默认 120 s），自动收集生成的图像文件
- **`toolkit.py`**：内置工具函数（`read_stream`、`filter_stream`、`plot_stream`、`compute_hvsr`、`estimate_magnitude_ml` 等）
- **`doc_parser.py`**：从 PDF 提取与代码任务相关的上下文片段

### `seismo_stats/` — 地震统计分析

- **`bvalue.py`**：完整性震级 Mc 估算（最大曲率法 / 拟合优度法）、b 值计算（MLE / LSQ）、σ_b 不确定性
- **`catalog_loader.py`**：地震目录加载，支持 CSV / JSON / picks.txt 多格式，自动识别列名
- **`plotting.py`**：F-M 分布图、时序活动图、震中分布图，Matplotlib 风格统一

### `seismo_tools/` — 外部工具注册表

统一管理 HypoDD、VELEST、HASH 等第三方地震学工具，支持：自动生成控制文件、调用外部可执行程序、解析输出结果，并通过对话指令触发。

### `web_app/rag_engine.py` — 知识库 RAG 引擎

| 环节 | 实现 |
|------|------|
| PDF 解析 | pdfminer.six（优先）/ PyMuPDF（兜底） |
| 文本分块 | 500 字/块，50 字重叠滑窗 |
| 向量化 | BGE-M3（1024 维，L2 归一化，支持中英文） |
| 索引 | FAISS IndexFlatIP（内积 = 余弦相似度） |
| 检索 | Top-K 召回 + 相似度阈值过滤 |
| 持久化 | `~/.seismicx/knowledge/`，重启后自动加载 |

---

## 目录结构

```
sage/
├── web_app/                    # Web 界面
│   ├── app.py                  # Flask 主应用（API 路由）
│   ├── rag_engine.py           # BGE-M3 + FAISS 知识库引擎
│   ├── requirements.txt        # Python 依赖
│   └── templates/
│       ├── chat.html           # 对话页面（marked.js 渲染）
│       ├── knowledge.html      # 知识库管理页
│       ├── skills.html         # 技能管理页（查看/编辑/新建）
│       └── llm_settings.html   # LLM 配置页
│
├── seismo_skill/               # 技能文档系统 ⚡
│   ├── skill_loader.py         # 解析、检索、注入（支持中英文混合检索）
│   ├── __init__.py             # 公开接口
│   ├── waveform_io.md          # 波形读取技能
│   ├── waveform_processing.md  # 波形预处理技能
│   ├── waveform_visualization.md  # 波形可视化技能
│   ├── spectral_analysis.md    # 频谱分析与 HVSR 技能
│   ├── b_value_analysis.md     # b 值统计分析技能
│   └── source_parameters.md   # 震源参数估算技能
│
├── seismo_agent/               # 自主 Agent
│   ├── agent_loop.py           # 主循环（SeismoAgent 类）
│   ├── planner.py              # 任务规划（TaskPlanner）
│   ├── memory.py               # 工作记忆（AgentMemory）
│   └── paper_reader.py         # 文献加载（load_paper）
│
├── seismo_code/                # 代码生成与执行
│   ├── code_engine.py          # LLM 代码生成（含技能注入）
│   ├── safe_executor.py        # 沙箱执行（execute_code）
│   ├── toolkit.py              # 内置地震学工具函数
│   └── doc_parser.py           # PDF 内容提取
│
├── seismo_stats/               # 地震统计分析
│   ├── bvalue.py               # b 值 / Mc 计算
│   ├── catalog_loader.py       # 地震目录加载
│   └── plotting.py             # 统计图绘制
│
├── seismo_tools/               # 外部工具注册表
│   └── tool_registry.py        # HypoDD / VELEST / HASH 等
│
├── pnsn/                       # 震相拾取
│   ├── picker.py               # 单台拾取脚本
│   ├── sage_picker.py          # 批量拾取（SagePicker 类）
│   ├── fastlinker.py           # FastLink 震相关联
│   ├── gammalink.py            # Gamma 震相关联
│   ├── pickers/                # JIT / ONNX 模型文件
│   └── config/                 # 拾取器参数配置
│
├── conversational_agent.py     # 对话 Agent 核心（意图分类 + 技能执行）
├── config_manager.py           # LLM 配置管理（统一读取 config.json）
├── backend_manager.py          # 多后端支持（Ollama / vLLM / 在线 API）
└── seismic_cli.py              # 命令行入口

~/.seismicx/                    # 用户数据目录（自动创建）
├── config.json                 # LLM 和工作区配置
├── knowledge/                  # 知识库向量索引（FAISS）
└── skills/                     # 用户自定义技能文档（.md）
```

---

## 配置文件

配置存储在 `~/.seismicx/config.json`，通过 LLM 设置页或 CLI 自动维护，无需手动编辑。

```json
{
  "llm": {
    "provider": "ollama",
    "model": "qwen3:30b",
    "api_base": "http://localhost:11434",
    "api_key": ""
  },
  "workspace": {
    "enabled": true,
    "path": "/data/seismic"
  }
}
```

| 字段 | 说明 |
|------|------|
| `llm.provider` | `ollama` / `openai` / `custom` |
| `llm.model` | 模型名称（Ollama tag 或 API 模型名） |
| `llm.api_base` | API 端点地址 |
| `llm.api_key` | 在线 API 密钥（Ollama 不需要） |
| `workspace.enabled` | 是否允许 LLM 访问本地文件列表 |
| `workspace.path` | 授权的根目录（LLM 无法访问此路径以外的内容） |

用户自定义技能存储在 `~/.seismicx/skills/`，每个 `.md` 文件为一个技能，格式见[技能文件格式](#创建自定义技能)。

---

## pnsn 震相拾取模块

pnsn 是一个专门用于震相拾取的深度学习模型库，由 Cangyu Chen 开发。该模块提供了多种先进的震相拾取模型，可以直接集成到 SAGE 中。

### 模型类型

1. **PhaseNet**：经典的 P/S 波初至拾取模型
2. **EQTransformer**：地震事件检测与震相拾取一体化模型
3. **JMA Picker**：基于日本气象厅算法的拾取器

### 使用方法

```bash
# 克隆仓库
git clone https://github.com/cangyeone/pnsn.git

# 进入目录
cd pnsn

# 安装依赖
pip install -r requirements.txt

# 运行批量拾取
python batch_pick.py --input_dir /path/to/data --output_dir /path/to/output
```

### 配置参数

在 SAGE 中使用 pnsn 模型时，可通过以下参数进行配置：

- `model_type`：选择使用的模型类型 (phasenet, eqtransformer, jma)
- `min_prob`：最小拾取概率阈值
- `filter_events`：是否过滤事件
- `output_format`：输出格式 (csv, xml, etc.)

### 性能优化

- 支持 JIT 编译加速
- 支持 ONNX 推理加速
- 支持 GPU/CPU 自适应选择

---

## 常见问题

**Q: 对话返回"当前没有配置可用的 LLM 模型"**

前往 `/llm-settings` 选择一个已安装的 Ollama 模型并点击「保存配置」。如使用在线 API，填写 Base URL、Key 和模型名后保存。

**Q: 知识库上传 PDF 后向量化很慢**

首次运行会从 HuggingFace 下载 BGE-M3 模型（约 2 GB），完成后速度正常。国内可设置镜像加速：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: Agent 步骤执行失败**

Agent 默认最多重试 2 次，失败步骤会跳过继续执行后续步骤。可增加 `--max-steps` 上限，或查看输出目录中的日志了解详情。

**Q: 批量拾取速度慢**

默认使用 CPU，添加 `--device cuda` 可启用 GPU 加速（需要 CUDA 环境和对应 PyTorch）。

**Q: 如何让 AI 使用我自己的函数库？**

在 `~/.seismicx/skills/` 下创建一个 `.md` 文件，按[技能文件格式](#创建自定义技能)写明函数签名、参数说明和最小示例。保存后无需重启，下一次对话或代码生成时即可自动匹配并参考。

**Q: 如何覆盖某个内置技能的说明？**

在 `~/.seismicx/skills/` 下创建与内置技能同名的文件（`name` 字段相同），用户版本会自动覆盖内置版本。

**Q: 如何添加新的地震学外部工具（如 HypoDD）？**

在 `seismo_tools/tool_registry.py` 中调用 `register_tool()` 注册工具的参数模板、输入文件格式和调用命令，随后可通过对话触发。也可在 `seismo_skill/` 中创建对应技能文档，描述输入文件格式和调用方式，让 AI 在代码生成时自动参考。

### RAG 功能无法使用

如果遇到"未找到嵌入模型库"的错误，尽管已经安装了相关库，请尝试以下步骤：

1. **确认库已安装**：
   ```bash
   pip list | grep -E "(FlagEmbedding|sentence-transformers)"
   ```

2. **版本兼容性问题**：
   有时不同版本的库之间存在冲突，尝试：
   ```bash
   pip install --upgrade FlagEmbedding sentence-transformers
   ```

3. **使用简化版RAG**：
   如果复杂依赖始终无法解决，项目包含一个简化版的向量数据库实现，可以在不依赖复杂嵌入模型的情况下运行基本的RAG功能。

4. **清理重装**：
   在某些情况下，完全卸载后再重装库可能有助于解决依赖问题：
   ```bash
   pip uninstall FlagEmbedding sentence-transformers transformers
   pip install FlagEmbedding sentence-transformers
   ```

