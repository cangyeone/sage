<p align="center">
  <img src="logo.png" alt="SeismicX logo" width="180"/>
</p>

<h1 align="center">SAGE — Seismology AI-Guided Engine</h1>

<p align="center">
  面向地震学研究的对话式 AI 分析平台 · Conversational AI Platform for Seismology Research
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-Flask-lightgrey" alt="Flask"/>
  <img src="https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI%20Compatible-green" alt="LLM"/>
  <img src="https://img.shields.io/badge/RAG-BGE--M3%20%2B%20FAISS-orange" alt="RAG"/>
  <img src="https://img.shields.io/badge/License-GPLv3-blue" alt="License"/>
</p>

---

SAGE 是集**自然语言交互**、**智能震相拾取**、**统计分析**、**代码生成执行**、**GMT 地图绘制**和**文献解读**于一体的地震学 AI 平台。用户通过中英文对话即可驱动完整分析流程，无需记忆命令行参数或编写样板代码。

---

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [安装](#安装)
  - [系统要求](#系统要求)
  - [基础安装](#基础安装)
  - [pnsn 震相拾取模块](#pnsn-震相拾取模块位置)
  - [RAG 功能依赖](#rag-功能依赖)
- [配置 LLM 后端](#配置-llm-后端)
- [Web 界面](#web-界面)
- [命令行工具](#命令行工具)
- [对话路由机制](#对话路由机制)
- [seismo_skill 技能系统](#seismo_skill-技能系统)
- [seismo_script 工作流系统](#seismo_script-工作流系统)
- [GMT 地图绘制](#gmt-地图绘制)
- [核心模块详解](#核心模块详解)
- [目录结构](#目录结构)
- [配置文件](#配置文件)
- [常见问题](#常见问题)

---

## 功能概览

| 模块 | 功能描述 |
|------|---------|
| 💬 **智能对话路由** | LLM 自动判断意图（知识问答 / 代码执行 / 闲聊），无需手动切换模式 |
| 🔍 **震相拾取** | 单台在线拾取 / 目录批量拾取，支持 JIT 与 ONNX 多种深度学习模型 |
| 🔗 **震相关联** | FastLink / REAL / Gamma 多方法，将台站拾取结果自动关联为地震事件 |
| 🧭 **极性分析** | P 波初动极性自动判断 |
| 📊 **地震统计** | b 值估算（MLE/LSQ）、F-M 分布图、时序与空间分布分析 |
| 🧑‍💻 **代码生成执行** | 内置 CodeEngine 负责沙箱 Python/GMT 科学脚本；Aider 作为仓库级代码后端用于修 bug、重构和多文件协同修改；OpenHands 可作为实验后端 |
| 🗺️ **GMT 地图绘制** | 调用 GMT6 绘制震中图、台站图、地形图、震源机制球，图像与脚本均可下载 |
| 🤖 **科学分析 Agent** | 读入数据、本地论文、Web 证据、RAG 和多个 SKILL → 规划科学问题 → 编程生成图表 → 迭代撰写 Markdown/LaTeX 论文 |
| 📚 **知识库 RAG** | BGE-M3 向量化 + FAISS 检索，持久化存储，批量 PDF 入库与文献问答 |
| 📖 **文献解读** | 临时上传 PDF → 深度解读方法/公式/结论，多轮追问 |
| 🗂 **本地文件访问** | 授权指定目录后，LLM 可直接读取文件列表辅助分析 |
| ⚡ **技能系统** | OpenAI-style 文件夹技能、内置领域技能、文档生成技能和内置学术研究技能；Chat、科学分析和 CodeEngine 可联合调用多个技能与 RAG |
| 🔄 **工作流系统** | 声明式多步分析流水线（`.md` + YAML frontmatter）；Agent 按步 DAG 调度 Code Engine 逐步执行，共享工作目录，每步独立 debug 循环 |
| 🎛 **参数优化 Agent** | Alpha 界面用于定义流程模块、输入输出、待优化参数和优化目标；CodeEngine 自动实现、调试、监控并保存优化过程，供科学分析写作使用 |
| 📈 **波形可视化** | 对话窗口内嵌波形图（震相标注叠加），图像可点击放大或下载 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI (Flask + JS)                          │
│   /chat · /science-analysis-agent · /parameter-optimization-agent│
│   /knowledge · /skills · /config                                 │
└──────────────┬──────────────────────────────────────────────────┘
               │ HTTP REST API
┌──────────────▼───────────────────────────────────────────────────┐
│   /api/chat/route（LLM 意图路由）  │  /api/chat/workflow          │
│      code ──────┬── qa ── chat    │   （工作流专用端点）           │
└────────┬────────┼────────┬────────┴────────────┬─────────────────┘
         │        │        │                      │
  ┌──────▼──────┐ ┌▼──────┐ ┌▼───────┐   ┌───────▼──────────┐
  │ CodeEngine  │ │RAG问答│ │通用对话│   │ CodeEngine       │
  │ + Toolkit   │ │BGE-M3 │ │        │   │ .run_workflow()  │
  │ + GMT       │ │+FAISS │ │        │   └───────┬──────────┘
  └──────┬──────┘ └───────┘ └────────┘           │
         │                               ┌────────▼───────────────────────┐
         │                               │  seismo_script 工作流调度器    │
         │                               │  步骤 DAG 拓扑排序 + 执行引擎  │
         │                               │  内置工作流 + ~/.seismicx/     │
         │                               │  workflows/                    │
         │                               └────────┬───────────────────────┘
         └──────────────────┬──────────────────────┘
                            │
  ┌─────────────────────────▼──────────────────────────────────────┐
  │            seismo_skill 技能检索                                │
  │    内置 7 个技能  +  用户自定义技能                              │
  │    (~/.seismicx/skills/)                                        │
  └─────────────────────────┬──────────────────────────────────────┘
                            │ 自动注入函数说明 + 代码示例
  ┌─────────────────────────▼──────────────────────────────────────┐
  │            LLM Backend                                          │
  │   Ollama（本地）  ·  vLLM  ·  OpenAI 兼容                      │
  └────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │           pnsn/ 震相拾取引擎                 │
  │    PhaseNet / EQTransformer / JIT / ONNX    │
  │    FastLink / Gamma 震相关联                 │
  └─────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 1. 克隆主仓库和内置子模块
git clone --recurse-submodules https://github.com/cangyeone/sage.git
cd sage

# 如果已经普通 clone 过，补拉子模块：
# git submodule update --init --recursive

# 2. 安装 SAGE 依赖
pip install -r requirements.txt

# 3. 启动 Ollama 并拉取模型（选一个）
ollama serve &
ollama pull qwen3:8b          # 轻量，约 6 GB

# 4. 启动 Web 服务
python web_app/app.py --port 5010

# 5. 浏览器访问
open http://localhost:5010
```

首次访问时，在 **LLM 设置页** 选择已拉取的模型并保存，即可开始使用所有功能。

---

## 安装

### 系统要求

| 资源 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | macOS / Linux / Windows | macOS 13+ / Ubuntu 22.04+ |
| **Python** | 3.9 | 3.10 / 3.11 |
| **内存 (RAM)** | 8 GB | 16 GB+（运行本地 LLM） |
| **存储空间** | 5 GB | 30 GB+（模型 + 知识库） |
| **GPU** | 可选 | CUDA 11.8+ 或 Apple Metal（加速推理） |

### 基础安装

```bash
git clone https://github.com/cangyeone/sage.git
cd sage

# 完整安装（推荐）
pip install -r requirements.txt

# 或按需安装各部分
pip install flask flask-cors                          # Web 服务
pip install obspy torch scipy numpy pandas            # 地震数据处理
pip install matplotlib plotly                         # 可视化
pip install FlagEmbedding faiss-cpu pdfminer.six PyMuPDF  # RAG 知识库
```

### pnsn 震相拾取模块位置

pnsn 是专门用于震相拾取的深度学习模型库，由 [cangyeone](https://github.com/cangyeone) 开发。在 SAGE 中，pnsn 作为 OpenAI-style 技能 `pnsn_phase_detection` 的组成部分管理，推荐位置为 `seismo_skill/skills/pnsn_phase_detection/pnsn/`，这样代码、配置和模型文件都跟随技能统一管理。

```bash
# 仅当技能目录下缺少 pnsn 时需要执行
git clone https://github.com/cangyeone/pnsn.git \
  seismo_skill/skills/pnsn_phase_detection/pnsn
```

当前 pnsn 仓库没有单独的 `requirements.txt`，用于 SAGE 时也不需要把 pnsn 安装成 Python 包。只需安装 SAGE 顶层依赖 `pip install -r requirements.txt`；SAGE 会直接调用 `seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py`、`fastlinker.py`、`pickers/*.jit` 等文件。

**目录结构确认：**

```
sage/
├── seismo_skill/
│   └── skills/
│       └── pnsn_phase_detection/
│           ├── SKILL.md
│           └── pnsn/       ← 技能内置 pnsn 代码和模型
│               ├── picker.py
│               ├── fastlinker.py
│               ├── gammalink.py
│               ├── pickers/  ← JIT / ONNX 模型文件
│               └── config/
├── web_app/
└── ...
```

pnsn 提供的主要模型：

| 模型 | 用途 | 格式 |
|------|------|------|
| **PhaseNet** | P/S 波到时拾取 | JIT / ONNX |
| **EQTransformer** | 事件检测 + 震相拾取一体化 | JIT / ONNX |
| **JMA Picker** | 基于日本气象厅算法的拾取器 | JIT |

### RAG 功能依赖

知识库 RAG 功能需要 `tokenizers` 库，后者在部分系统上需要 Rust 编译环境：

```bash
# 安装 Rust（仅在 pip install 报编译错误时需要）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 重新安装嵌入模型库
pip install FlagEmbedding sentence-transformers

# 首次使用时 BGE-M3 模型（约 2 GB）会自动从 HuggingFace 下载
# 国内网络可设置镜像：
export HF_ENDPOINT=https://hf-mirror.com
```

#### 国内用户推荐：使用 ModelScope 下载 BGE-M3

若 HuggingFace 无法访问，可先通过 ModelScope 将模型下载到本地：

```bash
pip install modelscope

python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/bge-m3', local_dir='open_models/bge-m3')
"
```

下载完成后，在 SAGE 中配置本地路径，使其直接读取本地模型而不联网下载。有两种方式：

**方式一 — Web 界面**（推荐）：  
打开**知识库页面**（`/knowledge`）→ 点击「嵌入模型」旁的 ⚙ 齿轮图标 → 粘贴绝对路径（如 `/Users/yourname/open_models/bge-m3`）→ 点击「保存」。

**方式二 — 直接编辑配置文件**：  
在 `~/.seismicx/config.json` 中添加 `embedding` 字段：

```json
{
  "llm": { "...": "..." },
  "embedding": {
    "model_path": "/Users/yourname/open_models/bge-m3"
  }
}
```

将 `model_path` 置为空字符串或删除该字段即可恢复为 HuggingFace 自动下载。该配置在下次文档构建时生效，**无需重启服务**。

---

## 配置 LLM 后端

所有 AI 功能均需要 LLM 后端。配置通过 **Web 界面 → LLM 设置页**，或命令行完成，统一存储在 `~/.seismicx/config.json`，修改后**立即生效，无需重启**。

### 方式一：Ollama（推荐，本地，无需联网）

```bash
# 1. 安装 Ollama
# macOS / Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# 或访问 https://ollama.ai/download

# 2. 启动服务
ollama serve

# 3. 拉取模型（根据显存 / 内存选择）
ollama pull qwen3:8b         # ~6 GB，适合日常使用
ollama pull qwen3:30b        # ~20 GB，综合能力强
ollama pull deepseek-r1:8b   # ~9 GB，推理能力强
ollama pull llama3.3:latest  # ~40 GB，英文能力强
```

在 LLM 设置页选择模型并点击「保存配置」即完成配置。

### 方式二：在线 API（OpenAI 兼容格式）

在 LLM 设置页 → 选择「自定义 API」并填写：

| 字段 | 示例（DeepSeek） | 示例（SiliconFlow） |
|------|----------------|-------------------|
| **API Base URL** | `https://api.deepseek.com/v1` | `https://api.siliconflow.cn/v1` |
| **API Key** | `sk-xxxxxxxx` | `sk-xxxxxxxx` |
| **模型名称** | `deepseek-chat` | `Qwen/Qwen2.5-72B-Instruct` |

支持任意 OpenAI 兼容接口，包括 DeepSeek、SiliconFlow、月之暗面（Moonshot）、阿里通义（DashScope）、智谱 GLM、Anthropic 等。

### 方式三：命令行配置

```bash
# Ollama 本地模型
python seismic_cli.py backend use ollama --model qwen3:30b

# 在线 API
python seismic_cli.py backend use online \
    --provider deepseek \
    --api-key sk-xxx \
    --model deepseek-chat

# 查看所有后端状态
python seismic_cli.py backend status

# 自动检测可用后端
python seismic_cli.py backend auto
```

---

## Web 界面

启动后访问 `http://localhost:5010`，包含四个主要页面。

### 🗨 对话页（/chat）

主交互界面。**无需切换模式** —— 系统通过 LLM 自动判断每条消息的意图，路由到最合适的处理器：

| 发送的内容 | 自动路由到 |
|-----------|----------|
| "什么是 Q-filter 算法？" | 知识问答（RAG 检索） |
| "帮我对 /data/wave.mseed 做 1-10 Hz 带通滤波并画图" | 代码生成执行 |
| "帮我用 GMT 绘制中国地形图" | GMT 技能执行 |
| "你好" | 通用对话 |

**侧边栏：**
- 📎 上传 PDF（当前会话临时使用）
- 🗂 授权本地工作目录（LLM 可读取指定路径的文件列表）
- 知识库文献数 / 片段数状态显示

**图像展示与下载：**
- 代码执行生成的图像直接嵌入对话气泡
- 每张图下方显示工具栏：**⬇ 图像** 下载 PNG，**⬇ GMT脚本** 下载可重现的 `.sh` 脚本（仅 GMT 图有）
- 点击图像可在新窗口全屏查看

**典型对话示例：**

```
# 知识问答（自动检索知识库）
> 什么是 Q-filter 算法？
> 解释一下 HVSR 谱比法的原理

# 数据处理（自动执行代码）
> 看下目录 /data/seismic/waveform 中的文件
> 绘制一下波形
> 对波形做 1-10 Hz 带通滤波后画图
> 计算垂向分量的功率谱密度

# GMT 地图
> 帮我用 GMT 绘制中国地形图
> 绘制 90-120°E、20-45°N 范围的震中分布图

# 文献解读
> 这篇论文的核心方法是什么？（上传 PDF 后提问）
```

### 📚 知识库页（/knowledge）

- 拖拽上传多个 PDF，自动使用 **BGE-M3** 向量化入库
- 实时显示索引进度（文本提取 → 分块 → 嵌入 → FAISS 写入）
- 文献管理：查看页数/片段数/文件大小，支持单篇删除或全量清空
- **持久化存储**：重启服务后知识库自动加载，无需重新上传

> 存储路径：`~/.seismicx/knowledge/`

### ⚡ 技能管理页（/skills）

无需重启即可扩展 AI 能力。页面包含**技能**和**工作流**两个标签页。

**技能标签页：**
- 左侧：内置技能（只读）与用户自定义技能（可编辑/删除）分组展示
- 右侧：Markdown 编辑器 + 实时预览，含语法高亮
- 支持新建、编辑、删除自定义技能
- 保存后下一次对话或代码生成立即生效

> 自定义技能存储路径：`~/.seismicx/skills/`

**工作流标签页：**
- 列出内置和用户自定义工作流，显示标题、版本和技能依赖徽章
- 步骤 DAG 预览面板：以节点 + 箭头图形可视化步骤依赖关系
- Markdown 编辑器，用于编辑 `.md` 工作流文件（YAML frontmatter + 流程说明体）
- 支持新建、编辑、删除自定义工作流

> 自定义工作流存储路径：`~/.seismicx/workflows/`

### ⚙️ LLM 设置页（/llm-settings）

- 在线检测 Ollama 已安装模型，一键选择
- 支持配置任意 OpenAI 兼容 API
- 保存后立即对所有功能生效
- 顶部徽章实时显示当前使用的模型

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
python seismic_cli.py pick \
    -i /data/station/ \
    -m seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.jit

# 批量拾取（目录下所有波形文件）
python seismic_cli.py pick \
    -i /data/seismic/2024/ \
    --batch \
    -o results/picks.csv

# 指定计算设备
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

SAGE 的编程能力分为两个互补后端：

- **内置 CodeEngine**：默认后端，适合科学数据脚本、GMT/Python 绘图、mini test 和可复现中间产物。
- **Aider 后端**：作为 SAGE 内部仓库级 Coding Backend，用于修复 bug、重构、多文件编辑和 Git 工作流。SAGE 会优先加载项目内置源码 `third_party/aider`，通过 Aider Python scripting API 调用，必要时退回已安装包或 CLI。安装项目依赖 `pip install -r requirements.txt` 后，在 `Config -> Coding Agent` 中选择 **Aider API / CLI**。
- **OpenHands 后端**：实验性 CLI 后端，适合更重的 agentic development 工作流。

`third_party/aider` 使用 Git submodule 管理。clone 时建议使用 `git clone --recurse-submodules ...`；如果已经普通 clone 过，则运行 `git submodule update --init --recursive`，这样本地才会真正拉下 Aider 源码。

代码后端配置写入项目目录下的 `seismo_rag/project_config.json`，便于检查、选择性纳入版本控制或清理。

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
python seismic_cli.py skill list                     # 列出所有技能
python seismic_cli.py skill search "带通滤波"        # 关键词搜索
python seismic_cli.py skill show waveform_processing # 查看完整文档
python seismic_cli.py skill new my_tool              # 新建自定义技能
python seismic_cli.py skill edit my_tool             # 编辑已有技能
python seismic_cli.py skill delete my_tool           # 删除技能
python seismic_cli.py skill dir                      # 查看技能目录路径
```

### LLM 后端管理

```bash
python seismic_cli.py backend status          # 查看当前状态
python seismic_cli.py backend setup           # 交互式配置向导
python seismic_cli.py backend auto            # 自动检测并选择
python seismic_cli.py backend models          # 列出本地已下载模型
python seismic_cli.py backend pull qwen3:8b   # 拉取 Ollama 模型
```

---

## 对话路由机制

SAGE 通过专用的 LLM 路由调用自动判断每条消息的意图，避免关键词误匹配（例如"Q-filter **algorithm**"不会被错误路由到代码执行）。

### 路由流程

```
用户消息
   │
   ├─ 快速路径：消息包含绝对路径（/data/...）且非问句
   │              └─→ code（直接执行）
   │
   └─ LLM 路由调用（max_tokens=10，约 <1s）
          │
          ├─ code  → CodeEngine 生成并执行 Python / GMT 代码
          ├─ qa    → RAG 检索知识库 + LLM 回答
          └─ chat  → 通用对话
```

### 三类路由说明

| 路由 | 触发条件 | 示例 |
|------|---------|------|
| `code` | 数据处理、绘图、文件操作、GMT 地图 | "对波形做带通滤波并画图"、"帮我用 GMT 绘制中国地形图" |
| `qa` | 概念解释、方法介绍、文献检索 | "什么是 Q-filter？"、"解释一下 HVSR 的原理" |
| `chat` | 打招呼、闲聊、非地震学内容 | "你好"、"今天天气怎么样" |

**LLM 不可用时的回退规则：**

- 消息含 `绘制/画图/滤波/频谱/waveform/.sac/.mseed` → `code`
- 其他 → `qa`

---

## seismo_skill 技能系统

技能系统是 SAGE 的核心扩展机制。每个技能是一个 Markdown 文档，描述函数用法和代码示例。**AI 对话和代码生成时自动检索并注入最相关的技能文档**，显著提升生成代码的准确性和规范性。

### 工作原理

```
用户消息（自然语言）
       │
       ▼
  seismo_skill 关键词检索
  （中英文混合 TF-IDF 评分）
       │
       ├─ 匹配到技能 → 将函数签名 + 示例代码注入 LLM 系统提示
       │
       ▼
  LLM 生成代码 / 回答
  （优先使用技能文档中的规范写法）
```

检索点已集成到：
- `/api/chat/rag`（Web 知识问答）
- `seismo_code/code_engine.py`（代码生成引擎）
- `seismo_agent/agent_loop.py`（自主 Agent 每步代码生成）

### 内置技能（7 个）

| 技能文件 | 类别 | 主要函数 |
|----------|------|---------|
| `waveform_io.md` | waveform | `read_stream`, `read_stream_from_dir`, `stream_info`, `picks_to_dict` |
| `waveform_processing.md` | waveform | `detrend_stream`, `taper_stream`, `filter_stream`, `resample_stream`, `trim_stream`, `remove_response` |
| `waveform_visualization.md` | visualization | `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion` |
| `spectral_analysis.md` | analysis | `compute_spectrum`, `compute_hvsr` |
| `b_value_analysis.md` | statistics | `load_catalog_file`, `calc_mc_*`, `calc_bvalue_mle`, `plot_gr` |
| `source_parameters.md` | analysis | `estimate_magnitude_ml`, `estimate_corner_freq`, `estimate_seismic_moment`, `moment_to_mw`, `estimate_stress_drop` |
| `gmt_plotting.md` | visualization | `run_gmt`（震中图、台站图、地形图、震源机制球、剖面图） |

### 创建自定义技能

**方式一：Web 界面**（推荐）

访问 `/skills` → 点击「新建自定义技能」→ 填写基本信息 → 在编辑器中完善文档。

**方式二：命令行**

```bash
python seismic_cli.py skill new my_hypodd_tool \
    --title "HypoDD 双差定位工具" \
    --keywords "双差定位, HypoDD, 精定位, relocation" \
    --desc "封装 HypoDD 输入文件生成和结果解析"
```

**方式三：直接编写 Markdown 文件**

在 `~/.seismicx/skills/` 下创建 `.md` 文件：

```markdown
---
name: my_skill_name
category: custom
keywords: 关键词1, 关键词2, english_keyword
related_skills:            # 可选 — 双向技能展开
  - waveform_io
  - tabular_io
workflow: seismicity_analysis   # 可选 — 关联的工作流名称
---

# 技能标题

## 描述

工具功能说明（一两句话）。

---

## 主要函数

### `function_name(param1, param2=default)`

**参数：**
- `param1` : type — 说明
- `param2` : type — 说明，默认 default

**返回：** type — 说明

```python
# 最小可运行示例
result = function_name("input", param2=42)
print(result)
```

---

## 注意事项

- 注意事项 1
```

> **覆盖规则：** 自定义技能与内置技能同名时，自定义版本自动优先生效。

### 从文档目录生成 OpenAI-style SKILL

SAGE 可以把外部文档转换为 OpenAI-style 文件夹型 SKILL。常见流程是：把文档目录放到 `seismo_skill/docs/`，在 Web 端知识库页面选择构建方式，然后由 Skill Builder 生成可复用技能到 `seismo_skill/user_skills/`。

#### 示例：从 GMT 中文文档生成 GMT SKILL

1. **下载 GMT 中文文档**

   访问 [gmt-china/GMT_docs releases](https://github.com/gmt-china/GMT_docs/releases)，下载一个 release 压缩包。建议选择包含源码文档、示例和资源文件的 release asset。

   也可以在终端下载，把 `<release-asset-url>` 替换为 release 页面中对应压缩包的下载地址：

   ```bash
   cd /path/to/sage
   mkdir -p seismo_skill/docs
   curl -L "<release-asset-url>" -o /tmp/GMT_docs.zip
   unzip /tmp/GMT_docs.zip -d seismo_skill/docs/
   ```

   确认最终目录类似：

   ```text
   seismo_skill/docs/GMT_docs-6.5/
     source/
     README.md
     ...
   ```

2. **启动 Web 服务**

   ```bash
   python web_app/app.py --port 5010
   ```

   浏览器打开 `http://localhost:5010/knowledge`。

3. **在 Web 端生成 SKILL**

   在 **技能文档目录** 卡片中：

   - 点击刷新，选择 `GMT_docs-6.5`。
   - 将 **SKILL 结构** 设为 **OpenAI-style 文件夹 SKILL**。
   - 如果文档很多，勾选 **RAG/向量辅助构建**。这里的 RAG 只用于构建阶段的相似度检索和聚类，最终产物仍然是 SKILL，不会把整个文档永久当作 RAG 文献。
   - **目标主题簇数** 可以留空，由系统自动建议；也可以手动填入希望的簇数。
   - 点击 **开始构建**。

4. **生成结果位置**

   构建完成后，生成的技能位于：

   ```text
   seismo_skill/user_skills/_gen_gmt_docs_zh/
     SKILL.md
     subskills/
     references/
     workflows/
     agents/
   ```

   其中 `SKILL.md` 是入口文件；`subskills/` 保存按功能聚类后的 GMT 子技能；`references/manifest.md` 记录参与构建的源文件，方便追溯。

5. **验证是否能自动调用**

   在 Chat 或 Code 页面直接提问：

   ```text
   GMT 的 -J 投影选项怎么用？给我几个常见投影示例。
   ```

   ```text
   用 GMT grdimage 绘制地形图，并添加 colorbar，解释参数。
   ```

   通常不需要显式写 `_gen_gmt_docs_zh`。只要问题中包含 `GMT`、`grdimage`、`makecpt`、`coast`、`-J`、`-R`、`-B` 等关键词，系统会自动把生成的 GMT 文档技能和内置 `gmt_plotting` 技能联合注入。

6. **管理和删除**

   生成型 SKILL 会在知识库/技能管理界面作为技能资产显示，可在界面中删除。删除时会同时移除生成的 SKILL 文件夹和构建元数据。

支持的文档输入包括 PDF、Markdown（`.md`）、reStructuredText（`.rst`）、HTML、纯文本、脚本文件，以及包含多种文件的混合文档目录。

#### 学术研究 SKILL

SAGE 已将 [academic-research-skills](https://github.com/Imbad0202/academic-research-skills) 集成到项目目录：

```text
third_party/academic-research-skills/
```

其中包含这些 OpenAI-style 技能：

- `deep-research`：面向文献证据的调研规划和证据综合
- `academic-paper`：学术论文写作、修改和结构化表达
- `academic-paper-reviewer`：模拟审稿人审查、提出修改意见
- `academic-pipeline`：端到端科研流程规划

可以通过后端接口安装或刷新到本地技能库：

```bash
curl -X POST http://localhost:5010/api/skills/install-academic-research \
  -H "Content-Type: application/json" \
  -d '{"overwrite": true}'
```

安装后，Chat、科学分析和 CodeEngine 都可以自动检索这些技能。一般不需要显式写技能名；例如“帮我做文献调研”“审稿式修改这篇论文”“写成 JGR 风格论文草稿”等请求，会自动把学术研究技能、本地地震学技能和 RAG 证据联合注入。

### 科学分析与参数优化

旧的地学解译页面和 `/api/evidence_geo_agent*` API 已移除。旧地址 `/evidence-geo-agent` 现在会自动跳转到科学分析页面：

```text
http://localhost:5010/science-analysis-agent
```

模块化优化流程请使用参数优化 Agent：

```text
http://localhost:5010/parameter-optimization-agent
```

参数优化 Agent 用于定义流程模块、每个模块的输入/输出、待优化参数和最终目标。Agent 会调用 CodeEngine 遍历项目目录、生成和调试脚本、运行 mini test、执行有边界的优化或 dry-run，并保存：

- `optimization_plan.md`
- `best_parameters.json`
- `optimization_history.csv`
- 图件和日志
- `optimization_report.md`

所有结果都保存在选择的项目目录内，后续科学分析 Agent 可以直接引用这些优化过程、图件和报告来撰写论文。

---

## seismo_script 工作流系统

工作流系统让你以声明式的 `.md` 文件定义多步分析流水线。每个工作流指定需要加载哪些技能、执行哪些步骤、步骤之间的依赖关系，由 Code Engine 负责代码生成与执行——工作流本身只充当调度蓝图。

### 角色分工

| 角色 | 职责 |
|------|------|
| **workflow（工作流）** | 作业流程书：执行哪些步骤、调用哪些技能、以何种顺序执行 |
| **skill（技能）** | 专项操作手册：如何使用某个具体工具或方法 |
| **agent（调度员）** | 匹配用户请求 → 加载对应工作流和技能 → 分解任务 |
| **code engine（程序员）** | 为每个步骤生成并修复 Python / GMT / Shell 代码 |
| **tool（工具）** | 执行器：Python 沙箱、GMT、Shell |

### 工作流文件格式

工作流使用与技能相同的 `.md` + YAML frontmatter 格式：

```markdown
---
name: seismicity_analysis
title: 地震活动性分析工作流
version: "1.0"
description: 完整的地震活动性分析，包括目录加载、时空分布和 b 值估算
keywords:
  - 地震活动性
  - b 值
  - 震中分布图
skills:
  - name: tabular_io
    role: 地震目录加载与解析
  - name: gmt_plotting
    role: 震中图绘制
  - name: b_value_analysis
    role: b 值估算与 GR 图绘制
steps:
  - id: load_catalog
    skill: tabular_io
    description: 从文件加载地震目录
  - id: epicenter_map
    skill: gmt_plotting
    description: 绘制震中分布图
    depends_on: [load_catalog]
  - id: b_value
    skill: b_value_analysis
    description: 计算 b 值并绘制 GR 分布
    depends_on: [load_catalog]
---

## 地震活动性分析流程说明

步骤一：使用 `load_catalog_file()` 加载目录...
```

**Frontmatter 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 工作流标识符 |
| `title` | str | 人类可读标题 |
| `description` | str | 单行摘要 |
| `keywords` | list[str] | 用于相关性搜索匹配 |
| `skills` | list[{name, role}] | 所需技能及各自的角色 |
| `steps` | list[{id, skill, description, depends_on}] | 执行 DAG |

Markdown 正文是**工作流指南**，在每个步骤的代码生成时注入到 LLM 上下文。

### 存储路径

| 位置 | 内容 |
|------|------|
| `seismo_script/workflows/` | 内置工作流（随 SAGE 发布） |
| `~/.seismicx/workflows/` | 用户自定义工作流（优先级更高，覆盖同名内置） |

### 内置工作流

| 工作流 | 说明 | 依赖技能 |
|--------|------|---------|
| `gmt_terrain_map` | GMT 地形图完整流水线（7 步：CPT → 裁剪 DEM → 渲染 → 海岸线 → 等高线 → 比例尺/图例 → 导出） | `gmt_plotting`, `_gen_gmt_docs_6_5` |
| `seismicity_analysis` | 地震活动性分析（目录 → 震中图 → 时序图 → b 值 → 剖面图） | `tabular_io`, `gmt_plotting`, `b_value_analysis` |

### `CodeEngine.run_workflow()` API

```python
result: WorkflowRunResult = engine.run_workflow(
    workflow_name    = "seismicity_analysis",
    user_request     = "分析 /data/catalog.csv 中的 2024 年地震目录",
    data_hint        = "/data/catalog.csv",   # 可选：注入步骤提示的路径提示
    max_debug_rounds = 3,                     # 每步失败时的最大重试次数
    timeout          = 120,                   # 每步执行超时（秒）
    skip_on_failure  = False,                 # True 时跳过失败步骤而非终止
    on_progress      = callback_fn,           # 可选：进度回调函数
)
```

`run_workflow()` 对步骤 DAG 进行拓扑排序，然后对每个步骤：
1. 检查所有 `depends_on` 前置步骤已成功完成
2. 扫描共享执行目录中已有的输出文件
3. 调用 `build_skill_context_with_rag()` 获取该步骤声明技能的上下文
4. 通过 LLM 生成代码（注入技能上下文 + 已完成步骤摘要）
5. 在共享目录中执行代码（步骤 N+1 可读取步骤 N 写入的文件）
6. 失败时：将错误文本追加到 RAG 查询并重试，最多 `max_debug_rounds` 次
7. 记录 `StepResult` 并追加到共享对话历史

**`WorkflowRunResult`：**

```python
@dataclass
class WorkflowRunResult:
    workflow_name: str
    steps:         List[StepResult]   # 每个执行步骤对应一条记录
    shared_dir:    str                # 所有步骤输出文件所在目录
    total_time:    float              # 总耗时（秒）

    @property
    def failed_steps(self)  -> List[StepResult]: ...
    @property
    def skipped_steps(self) -> List[StepResult]: ...
```

**`StepResult`：**

```python
@dataclass
class StepResult:
    step_id:      str
    skill:        str
    description:  str
    success:      bool
    code:         str
    stdout:       str = ""
    stderr:       str = ""
    figures:      List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    attempts:     int = 1
    diagnosis:    str = ""
    skipped:      bool = False
```

### Web API

**触发工作流运行：**

```
POST /api/chat/workflow
Content-Type: application/json

{
  "workflow_name":   "seismicity_analysis",
  "message":         "分析 /data/catalog.csv 中的 2024 年四川地震目录",
  "session_id":      "可选会话ID",
  "data_hint":       "/data/catalog.csv",
  "skip_on_failure": false
}

响应：{ "ok": true, "job_id": "wf_xxxx" }
```

**轮询结果**（与单步代码任务共用同一端点）：

```
GET /api/chat/code/poll/<job_id>

响应（已完成）：
{
  "status": "completed",
  "result": {
    "step_results": [
      { "step_id": "load_catalog",  "success": true,  "figures": [...], "stdout": "..." },
      { "step_id": "epicenter_map", "success": true,  "figures": ["/path/map.png"] },
      { "step_id": "b_value",       "success": false, "diagnosis": "Mc 过高", "attempts": 3 }
    ],
    "shared_dir": "/tmp/sage_wf_xxxxx"
  }
}
```

### 创建自定义工作流

**方式一：Web 界面**（推荐）

访问 `/skills` → **工作流**标签页 → 点击「新建工作流」→ 填写元数据 → 在编辑器中完善 Markdown 流程说明。步骤 DAG 预览会随 frontmatter 编辑实时更新。

**方式二：直接编写 `.md` 文件**

将文件保存到 `~/.seismicx/workflows/<name>.md`，使用上面展示的 frontmatter 格式。无需重启，文件立即被加载。

---

## GMT 地图绘制

SAGE 通过 `run_gmt()` 工具函数直接调用 GMT6，生成专业级地震学地图。

### 安装 GMT

```bash
# macOS
brew install gmt

# Linux（Conda 环境）
conda install -c conda-forge gmt

# Linux（apt）
sudo apt install gmt
```

### 使用方式

在对话中直接描述需求，SAGE 自动生成并执行 GMT 脚本：

```
> 帮我用 GMT 绘制中国地形图
> 绘制 90-120°E、20-45°N 的震中分布图
> 用 GMT 绘制台站分布图，数据在 /data/stations.txt
```

或在代码中调用（`run_gmt` 已预注入，无需 import）：

```python
gmt_script = """
gmt begin china_topo PNG
  gmt grdcut @earth_relief_01m -R70/140/15/55 -Gtopo.grd
  gmt grdimage topo.grd -JM16c -Cetopo1 -I+d
  gmt coast -W0.5p,gray40 -N1/0.8p -Baf -BWSne+t"中国地形图"
  gmt colorbar -DJBC+w8c/0.4c -Baf+l"Elevation (m)"
gmt end
"""

run_gmt(gmt_script, outname="china_topo", title="中国地形图")
```

### 中文标题自动处理

GMT 的 PostScript 引擎不支持 CJK 字符。SAGE 自动处理这一问题：
1. 执行前从脚本中提取中文标题/标签
2. 用空占位符替换，由 GMT 无乱码地渲染地图内容
3. 执行完成后，用 matplotlib 将中文标题叠加回 PNG

> **用户无需关心此细节**，直接在脚本里写中文标题即可。

### 图像与脚本下载

每张 GMT 图像下方的工具栏提供：
- **⬇ 图像**：下载 PNG 文件
- **⬇ GMT脚本**：下载 `.sh` 脚本文件，可在终端独立运行完整复现地图

---

## 核心模块详解

### `seismo_script/` — 工作流系统

```
seismo_script/
├── workflow_runner.py  # 工作流发现、搜索、CRUD 与上下文构建
├── workflows/          # 内置工作流 .md 文件（gmt_terrain_map、seismicity_analysis 等）
└── __init__.py         # 公开 API：list_workflows, search_workflows, load_workflow,
                        #   save_user_workflow, delete_user_workflow, build_workflow_context
```

**公开 API 一览：**

| 函数 | 说明 |
|------|------|
| `list_workflows()` | 返回所有工作流元数据（不含指南正文） |
| `search_workflows(query, top_k)` | 按关键词相关性对工作流排序 |
| `load_workflow(name)` | 返回完整工作流条目（含指南文本） |
| `save_user_workflow(name, text)` | 将 `.md` 文件保存到 `~/.seismicx/workflows/` |
| `delete_user_workflow(name)` | 删除用户自定义工作流 |
| `build_workflow_context(query)` | 返回 `(context_str, skill_names)` 供 LLM 注入 |

### `seismo_code/` — 代码生成与执行引擎

```
seismo_code/
├── code_engine.py      # LLM 代码生成（含技能注入、多轮历史、错误重试、
│                       #   run_workflow() 多步 DAG 执行）
├── safe_executor.py    # 沙箱执行（独立子进程、120s 超时、自动收集图像）
├── toolkit.py          # 内置地震学工具函数（无需 import，直接调用）
└── doc_parser.py       # 从 PDF 提取与代码任务相关的上下文片段
```

**内置工具包（`toolkit.py`，代码执行时自动注入）：**

| 类别 | 函数 |
|------|------|
| 数据读取 | `read_stream`, `read_stream_from_dir` |
| 波形处理 | `detrend_stream`, `taper_stream`, `filter_stream`, `resample_stream`, `trim_stream`, `remove_response` |
| 可视化 | `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion`, `plot_travel_time_curve` |
| 走时计算 | `taup_arrivals`, `p_travel_time`, `s_travel_time` |
| 频谱分析 | `compute_spectrum`, `compute_hvsr` |
| 震源参数 | `estimate_magnitude_ml`, `estimate_corner_freq`, `estimate_seismic_moment`, `moment_to_mw`, `estimate_stress_drop` |
| GMT 绘图 | `run_gmt` |
| 工具函数 | `stream_info`, `picks_to_dict`, `savefig` |

**沙箱执行机制：**
- 代码在独立子进程中运行，主进程不受崩溃影响
- 超时保护（默认 120 秒）
- 生成的图像通过 `[FIGURE] /path` 标记自动收集并发送到前端
- GMT 脚本通过 `[GMT_SCRIPT] /path` 标记单独收集，供前端提供下载

### `seismo_agent/` — 自主 Agent

从文献到代码的全自动实现流程：

```
seismo_agent/
├── paper_reader.py   # 文献加载（PDF / arXiv ID / DOI / 纯文本）
├── memory.py         # 跨步骤工作记忆（文献内容、步骤结果、已生成变量）
├── planner.py        # LLM 任务规划（目标 + 文献摘要 → JSON 步骤列表）
└── agent_loop.py     # 主循环（规划 → 代码 → 执行 → 失败重试 → 汇总）
```

执行流程：

```
用户目标 + 文献来源（PDF / arXiv / DOI）
       │
  加载并提取文献核心内容
       │
  LLM 规划执行步骤（3–8 步，JSON 格式）
       │
  ┌─── 每一步 ───────────────────────────┐
  │  检索相关技能文档（seismo_skill）     │
  │  LLM 生成代码（技能上下文注入）      │  ← 失败最多重试 2 次
  │  沙箱安全执行                        │
  │  记录结果和生成图像                   │
  └──────────────────────────────────────┘
       │ 循环所有步骤
  汇总报告 + 输出目录
```

### `web_app/rag_engine.py` — 知识库 RAG 引擎

| 环节 | 实现 |
|------|------|
| PDF 解析 | pdfminer.six（优先）/ PyMuPDF（兜底） |
| 文本分块 | 500 字/块，50 字滑窗重叠 |
| 向量化 | BGE-M3（1024 维，L2 归一化，中英文双语） |
| 索引 | FAISS `IndexFlatIP`（内积 = 余弦相似度） |
| 检索 | Top-K 召回 + 相似度阈值过滤，只显示真正命中的文献 |
| 持久化 | `~/.seismicx/knowledge/`，启动时自动加载；启动时自动清理已删除文件的孤立向量 |
| 回退 | BGE-M3 不可用时自动降级为 TF-IDF 余弦相似度检索 |

### `seismo_stats/` — 地震统计分析

```
seismo_stats/
├── bvalue.py         # Mc（最大曲率法 / 拟合优度法）+ b 值（MLE / LSQ）+ σ_b 不确定性
├── catalog_loader.py # 目录加载：CSV / JSON / picks.txt，自动识别列名
└── plotting.py       # F-M 分布图、时序活动图、震中分布图
```

### `seismo_tools/` — 外部工具注册表

统一管理 HypoDD、VELEST、HASH 等第三方地震学工具。支持自动生成控制文件、调用外部可执行程序、解析输出结果，可通过对话指令触发。

---

## 目录结构

```
sage/
├── web_app/                      # Web 服务
│   ├── app.py                    # Flask 主应用（40+ API 路由）
│   ├── rag_engine.py             # BGE-M3 + FAISS 知识库引擎
│   ├── simple_rag.py             # TF-IDF 回退 RAG
│   ├── simple_vector_db.py       # 轻量向量数据库（pickle 持久化）
│   └── templates/
│       ├── chat.html             # 对话页面（主界面）
│       ├── knowledge.html        # 知识库管理
│       ├── skills.html           # 技能管理
│       └── llm_settings.html     # LLM 配置
│
├── seismo_skill/                 # 技能文档系统
│   ├── skill_loader.py           # 解析、检索、注入（中英文混合检索）
│   ├── __init__.py
│   ├── waveform_io.md            # 波形读取
│   ├── waveform_processing.md    # 波形预处理
│   ├── waveform_visualization.md # 波形可视化
│   ├── spectral_analysis.md      # 频谱分析与 HVSR
│   ├── b_value_analysis.md       # b 值统计分析
│   ├── source_parameters.md      # 震源参数估算
│   ├── tabular_io.md             # CSV / TXT 数据读取
│   └── gmt_plotting.md           # GMT 地图绘制
│
├── seismo_script/                # 工作流系统
│   ├── workflow_runner.py        # 工作流发现、搜索、CRUD、上下文构建
│   ├── workflows/                # 内置工作流 .md 文件
│   │   ├── gmt_terrain_map.md    # GMT 地形图 7 步流水线
│   │   └── seismicity_analysis.md # 地震活动性分析流水线
│   └── __init__.py
│
├── seismo_code/                  # 代码生成与执行引擎
│   ├── code_engine.py            # LLM 代码生成（多轮历史 + 错误重试
│   │                             #   + run_workflow() DAG 执行）
│   ├── safe_executor.py          # 沙箱执行（子进程 + 超时保护）
│   ├── toolkit.py                # 内置地震学工具函数
│   └── doc_parser.py             # PDF 内容提取
│
├── seismo_agent/                 # 自主 Agent
│   ├── agent_loop.py             # 主循环（SeismoAgent 类）
│   ├── planner.py                # 任务规划（TaskPlanner）
│   ├── memory.py                 # 工作记忆（AgentMemory）
│   └── paper_reader.py           # 文献加载（load_paper）
│
├── seismo_stats/                 # 地震统计分析
│   ├── bvalue.py                 # b 值 / Mc 计算
│   ├── catalog_loader.py         # 地震目录加载
│   └── plotting.py               # 统计图绘制
│
├── seismo_tools/                 # 外部工具注册表
│   └── tool_registry.py          # HypoDD / VELEST / HASH 等
│
├── seismo_skill/
│   └── skills/
│       └── pnsn_phase_detection/ # OpenAI-style PNSN 震相拾取技能
│           ├── SKILL.md
│           └── pnsn/             # 技能内置 pnsn 代码和模型
│               ├── picker.py
│               ├── fastlinker.py
│               ├── gammalink.py
│               ├── pickers/      # JIT / ONNX 模型文件
│               └── config/       # 拾取器参数配置
│
├── conversational_agent.py       # 对话 Agent 核心（意图分类 + 技能执行）
├── config_manager.py             # LLM 配置管理
├── backend_manager.py            # 多后端支持（Ollama / vLLM / 在线 API）
├── seismic_cli.py                # 命令行入口
├── requirements.txt              # Python 依赖
└── logo.png

~/.seismicx/                      # 用户数据目录（首次运行自动创建）
├── config.json                   # LLM 和工作区配置
├── knowledge/                    # 知识库向量索引（FAISS + 元数据）
│   ├── faiss_index.bin
│   ├── metadata.json
│   └── pdfs/                     # PDF 副本
├── skills/                       # 用户自定义技能文档
│   └── my_custom_skill.md
└── workflows/                    # 用户自定义工作流 .md 文件（覆盖同名内置工作流）
    └── my_custom_workflow.md
```

---

## 配置文件

配置统一存储在 `~/.seismicx/config.json`，通过 Web 界面或 CLI 自动维护，无需手动编辑。

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

| 字段 | 说明 | 可选值 |
|------|------|--------|
| `llm.provider` | LLM 提供商 | `ollama` / `openai` / `custom` |
| `llm.model` | 模型名称 | Ollama tag 或 API 模型名 |
| `llm.api_base` | API 端点地址 | `http://localhost:11434`（Ollama 默认） |
| `llm.api_key` | API 密钥 | Ollama 无需填写 |
| `workspace.enabled` | 是否允许 LLM 访问本地文件列表 | `true` / `false` |
| `workspace.path` | 授权根目录（LLM 无法访问此路径以外的内容） | 绝对路径字符串 |

---

## 常见问题

**Q: 对话返回"当前没有配置可用的 LLM 模型"**

前往 `/llm-settings` 选择一个已安装的 Ollama 模型，或配置在线 API 后点击「保存配置」。

**Q: "what is filter algorithm?" 这类英文提问被错误路由到代码执行**

已修复。SAGE 使用 LLM 而非关键词正则来判断意图，概念类问句（含 filter、spectrum 等技术词）会被正确路由到知识问答，而非代码执行。

**Q: 知识库上传 PDF 后向量化很慢**

首次运行会从 HuggingFace 下载 BGE-M3 模型（约 2 GB）。完成后速度正常。国内网络可设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果 HuggingFace 完全无法访问，可使用 ModelScope 将模型下载到本地（参见[国内用户推荐：使用 ModelScope 下载 BGE-M3](#国内用户推荐使用-modelscope-下载-bge-m3)），然后在知识库页面的嵌入模型设置中配置本地路径。

**Q: GMT 图中的中文标题显示乱码**

无需特别处理。SAGE 已内置 CJK 自动处理：GMT 执行阶段用空占位符替换中文，执行完成后由 matplotlib 将中文标题叠加回 PNG，确保中文正确显示。

**Q: GMT 绘图失败，提示"GMT 未安装"**

安装 GMT >= 6.0：

```bash
# macOS
brew install gmt

# Linux（conda 环境）
conda install -c conda-forge gmt
```

**Q: 批量拾取速度慢**

默认使用 CPU。添加 `--device cuda` 启用 GPU 加速（需要 CUDA 环境及对应版本的 PyTorch）。

**Q: Agent 步骤执行失败**

Agent 默认每步最多重试 2 次，失败步骤会跳过并继续执行后续步骤。可增加 `--max-steps` 上限，或查看输出目录中的日志了解详情。

**Q: 如何让 AI 使用我自己的函数库？**

在 `~/.seismicx/skills/` 下创建一个 `.md` 技能文件，按[技能文件格式](#创建自定义技能)写明函数签名、参数说明和最小示例。保存后无需重启，下一次对话立即生效。

**Q: RAG 功能报错"未找到嵌入模型库"**

```bash
# 1. 确认已安装
pip list | grep -E "(FlagEmbedding|sentence-transformers)"

# 2. 尝试升级
pip install --upgrade FlagEmbedding sentence-transformers

# 3. 如需要 Rust 编译器
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
pip install FlagEmbedding sentence-transformers
```

如果以上方法均无法解决，项目内置的轻量 TF-IDF 向量数据库会自动作为回退方案，基本 RAG 功能仍然可用。

**Q: 如何添加 HypoDD 等外部工具的 AI 支持？**

在 `seismo_tools/tool_registry.py` 中调用 `register_tool()` 注册工具的参数模板和调用命令；同时在 `seismo_skill/` 中创建对应技能文档，描述输入文件格式，让 AI 在代码生成时自动参考。

---

<p align="center">
  <sub>Built with ❤️ for the seismology community</sub>
</p>

## 致谢

SAGE 的 Aider 集成后端基于开源项目 [Aider](https://github.com/Aider-AI/aider)，它为终端和 Git 工作流提供 AI 结对编程能力。SAGE 已将 Aider 源码放在 `third_party/aider`，优先通过 Python scripting API 集成，并保留已安装包和 CLI 退路以增强兼容性。实验性的 OpenHands 后端面向 [OpenHands](https://github.com/OpenHands/OpenHands) 的 CLI 工作流。感谢这些开源社区为更强的 coding-agent 能力提供基础。

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- **蔡育埼** - caiyuqiming@foxmail.com
- **刘鑫** - xinliu_geo@outlook.com
- **于子叶** - yuziye@cea-igp.ac.cn
## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 许可证。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
