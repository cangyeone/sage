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

SAGE 是一个 **Web 优先** 的地震学与地球物理 AI 工作台。它把对话式问答、科学分析、参数优化、知识库检索、OpenAI-style SKILL、代码执行、GMT/Python 绘图和论文写作组织到同一个界面中。用户可以上传数据、论文和说明文档，让系统自动判断文件作用、规划科学问题、编程生成图表、整合证据，并输出可复现的报告、Markdown 论文和 LaTeX 草稿。

命令行工具 `seismic_cli.py` 仍然保留，但现在主要作为脚本化、批处理和调试入口；日常使用推荐通过 Web 端完成。

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
- [命令行工具（高级/备用）](#命令行工具高级备用)
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

| 模块 | 当前定位 | 核心能力 |
|------|---------|---------|
| **Chat** | 日常入口 | 流式对话、PDF 临时解读、RAG 问答、Web search、图片/表格理解、代码执行、GMT/Python 绘图、多个 SKILL 联合调用 |
| **科学分析 Agent** | 科研主入口 | 遍历项目目录，识别数据/文献/说明，检索本地与在线文献，提出科学问题，规划图表，调用 CodeEngine 统计绘图，迭代写 Markdown/LaTeX 论文 |
| **参数优化 Agent** | 流程与模型优化入口 | 用户定义模块、输入输出、参数和目标；LLM 理解流程，CodeEngine 实现脚本，监控优化过程，结果可被科学分析引用写作 |
| **知识库** | 持久知识入口 | PDF/Markdown/项目/对话入库，BGE-M3 或 TF-IDF fallback，向量检索 + 关键词检索，支持删除和增量更新 |
| **技能系统** | 能力扩展入口 | 支持 OpenAI-style 文件夹 SKILL、内置 SKILL、文档生成 SKILL、学术研究 SKILL；Chat、科学分析、参数优化和 CodeEngine 均可调用 |
| **CodeEngine** | 执行与调试核心 | 生成 Python/GMT/Bash 脚本，运行 mini test，自我 debug，保存图件、表格、日志和中间产物；可借鉴 Aider 式多文件协同调试 |
| **LLM/Config** | 全局配置入口 | 配置 Ollama、本地模型、在线 API、OpenAI-compatible API、Web search 源、工作目录、代码后端和多模态能力 |
| **地震学工具链** | 领域能力 | 震相拾取、震相关联、极性分析、b 值统计、波形处理、GMT 地图、三维地形/速度结构绘图 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Web UI                                      │
│ /chat  /science-analysis-agent  /parameter-optimization-agent         │
│ /knowledge  /skills  /config                                         │
└───────────────┬──────────────────────────────────────────────────────┘
                │ REST + SSE streaming
┌───────────────▼──────────────────────────────────────────────────────┐
│                         Agent Orchestration                          │
│ intent routing · project isolation · background jobs · stop/resume    │
│ evidence tracking · reviewer-style iteration · multilingual prompts   │
└───────┬───────────────┬──────────────────┬──────────────────────────┘
        │               │                  │
┌───────▼──────┐ ┌──────▼──────┐  ┌────────▼────────┐
│ CodeEngine   │ │ RAG/Search  │  │ Skill Loader    │
│ Python/GMT   │ │ BGE-M3/FAISS│  │ OpenAI-style    │
│ Bash/LaTeX   │ │ keywords    │  │ built-in/user   │
│ mini tests   │ │ OpenAlex... │  │ nested subskills│
└───────┬──────┘ └──────┬──────┘  └────────┬────────┘
        │               │                  │
┌───────▼───────────────▼──────────────────▼──────────────────────────┐
│                         LLM Backends                                  │
│ Ollama · OpenAI compatible APIs · DeepSeek · SiliconFlow · DashScope  │
│ optional multimodal models for figure/table/image analysis            │
└───────┬──────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────────┐
│                         Domain Toolkits                               │
│ pnsn phase picking · GMT · ObsPy · statistics · document extraction   │
│ science paper templates · parameter optimization workflows            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 1. 克隆主仓库和内置子模块
git clone --recurse-submodules https://github.com/cangyeone/sage.git
cd sage

# 如果已经普通 clone 过，补拉子模块：
# git submodule update --init --recursive

# 2. 一键安装依赖、执行基础设置、后台启动 Web 服务
chmod +x sagectl.sh
./sagectl.sh
```

默认访问地址为 `http://127.0.0.1:5010`。首次访问时，在 **Config 页面** 选择或填写模型并保存，即可开始使用所有功能。

常用控制命令：

```bash
./sagectl.sh status    # 查看网站、端口、日志位置
./sagectl.sh logs      # 实时查看后台日志
./sagectl.sh stop      # 停止后台网站
./sagectl.sh start     # 再次后台启动
./sagectl.sh restart   # 重启
```

如需换端口：

```bash
SAGE_PORT=5011 ./sagectl.sh start
```

如果使用本地 Ollama，可先启动并拉取模型：

```bash
./sagectl.sh ollama-start
ollama pull qwen3:8b
```

备用手动方式：

```bash
pip install -r requirements.txt
python web_app/app.py --port 5010
```

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

所有 AI 功能均需要 LLM 后端。推荐通过 **Web 界面 → Config 页面** 配置本地模型、在线 API、Web search 源、代码后端和工作目录。项目相关配置优先写入项目目录，便于清理和迁移；少量用户级默认项由系统自动维护。

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

在 Config 页面选择模型并点击「保存配置」即完成配置。

### 方式二：在线 API（OpenAI 兼容格式）

在 Config 页面 → 选择「自定义 API」并填写：

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

启动后访问 `http://127.0.0.1:5010`。当前推荐把 Web 端作为主入口；它支持后台任务、实时流式输出、对话/项目隔离、文件持久化、图表渲染和多语言界面。

### Chat（/chat）

用于日常问答、论文解读、快速绘图和小规模数据处理。Chat 会根据上下文决定走 QA、RAG、Web search、SKILL 或 CodeEngine，执行过程中可以切换页面，回来后继续看到累积输出。

| 任务 | 示例 |
|------|------|
| 论文问答 | 上传 PDF 后问“这篇文章的核心方法和公式是什么？” |
| 数据处理 | “对这个 mseed 做 1-10 Hz 带通滤波并标出震相” |
| GMT/Python 绘图 | “用 GMT 绘制中国地形图，在中心加红色五角星” |
| Web research | “检索当前地震学大模型有哪些，并列出来源” |
| SKILL 联合调用 | “参考 GMT 文档技能和地形三维技能，绘制四川三维地形图” |

Chat 支持临时上传文件，也可以把对话、项目或文献显式加入知识库；未加入知识库的聊天文件默认只服务当前会话/项目。

### 科学分析（/science-analysis-agent）

面向“给定数据和文献，自动形成科研分析”的主页面。推荐把一个研究任务的所有数据、论文、说明、脚本和模板放在同一个项目目录中，Agent 会递归遍历并自动判断文件角色。

核心流程：

1. 识别数据、字段说明、论文、已有图表和 LaTeX/Markdown 模板。
2. 结合本地文献、知识库、Web search 和 SKILL，提出可验证科学问题。
3. 由 LLM 规划论文需要的图件、表格、统计量和反证路径。
4. 调用 CodeEngine 编程统计、绘图、生成中间产物，并在失败时自动 debug。
5. 依据图表和证据撰写 Markdown 论文，必要时同步生成 LaTeX/PDF。
6. 模拟严格审稿人进行多轮自评，补图、删图、改结论，直到问题收敛。

科学分析不是简单的数据质量汇总，而是尽量围绕“科学问题—证据—图表—论文结论”迭代。

### 参数优化（/parameter-optimization-agent）

用于定义可优化流程，例如震相检测模型训练、信号处理参数搜索、反演流程参数调优或自定义科学计算流水线。用户定义输入、输出、参数和目标函数；LLM 负责理解流程，CodeEngine 负责实现、运行、调试和监控。优化记录、图件和结果可以作为科学分析项目的材料。

### 知识库（/knowledge）

知识库用于长期可检索材料：论文 PDF、Markdown 文档、Chat/Project 导出内容、科学分析项目，以及由文档生成的 SKILL/RAG 辅助索引。检索采用向量检索与关键词检索结合；中文场景可使用 jieba 分词，向量模型优先使用 BGE-M3，缺失时降级到轻量 fallback。

知识库支持增量更新和删除。删除时会同步清理对应 RAG 内容、项目条目和生成的 SKILL 关联信息。

### 技能管理（/skills）

技能系统采用 OpenAI-style 文件夹结构，支持内置技能和 `seismo_skill/user_skills/` 下的用户技能。文档目录 `seismo_skill/docs/` 中的单个文件或文件夹可以通过 Web 端转换成技能；对于大规模文档，可先用 RAG/向量聚类辅助，把相似内容合并成同一个技能包下的 `subskills/`，再由 LLM 标准化成可复用说明、示例和约束。

技能可以被 Chat、科学分析、参数优化和 CodeEngine 联合调用。生成失败或不再需要的技能可以在界面删除。

### Config（/config）

Config 页面统一管理模型和系统能力：

- Ollama、本地模型、在线 OpenAI-compatible API。
- DeepSeek、SiliconFlow、DashScope、Moonshot/Kimi、Zhipu、自定义 API。
- Web search 源，例如 OpenAlex、Semantic Scholar、arXiv 和自定义搜索服务。
- Chat/Agent 工作目录、额外授权目录、代码后端和多模态能力。
- 是否显示/折叠 thinking、是否启用 RAG、Web search、图像表格解析等能力。

---

## 命令行工具（高级/备用）

`seismic_cli.py` 仍然可用，但现在定位为脚本化、批处理和调试入口。日常交互、科学分析、参数优化、知识库和技能管理建议优先使用 Web 端。

常用命令保留如下：

```bash
# 查看或自动选择模型后端
python seismic_cli.py backend status
python seismic_cli.py backend auto

# 对话/代码执行的轻量入口
python seismic_cli.py chat
python seismic_cli.py run "对 /data/wave.mseed 做 1-10 Hz 带通滤波并画图"

# 震相拾取和关联批处理
python seismic_cli.py pick -i /data/seismic/2024/ --batch -o results/picks.csv
python seismic_cli.py associate -i results/picks.csv -s station_list.csv --method fastlink -o results/events.txt

# 地震统计脚本
python seismic_cli.py stats bvalue -i catalog.csv --mc auto
python seismic_cli.py stats report -i catalog.csv
```

网站服务本身推荐使用根目录脚本控制：

```bash
./sagectl.sh          # 安装依赖、基础设置、后台启动
./sagectl.sh status   # 查看状态
./sagectl.sh logs     # 查看日志
./sagectl.sh stop     # 停止后台网站
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

### 内置技能

内置技能已统一为 OpenAI-style 文件夹结构：每个技能目录包含 `SKILL.md`，可选包含 `agents/`、`references/`、`assets/`、`workflows/` 和 `subskills/`。

| 技能目录 | 主要用途 |
|----------|---------|
| `waveform_io/` | 波形文件读取、目录扫描、元数据整理 |
| `waveform_processing/` | 去均值、去趋势、滤波、重采样、去仪器响应 |
| `waveform_visualization/` | 波形图、频谱图、PSD、粒子运动图 |
| `spectral_analysis/` | 频谱、HVSR、谱比和相关频域分析 |
| `b_value_analysis/` | b 值、完备震级、G-R 关系和地震活动性统计 |
| `source_parameters/` | 震级、角频率、地震矩、矩震级、应力降估计 |
| `gmt_plotting/` | GMT 地图、地形图、震中图、剖面和机制球绘制 |
| `terrain_3d_plotting/` | Python/Plotly/Three.js 风格的三维地形可视化 |
| `pnsn_phase_detection/` | PhaseNet/EQTransformer 震相拾取与监测流程 |
| `tabular_io/` | CSV/Excel/文本表格读取和字段推断 |
| `cartopy_plotting/` | Cartopy 地图绘制备用技能 |
| `nature-figure/`, `nature-data/`, `nature-polishing/` | 学术图件、数据整理和论文润色辅助 |

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

**方式三：直接编写 OpenAI-style 技能文件夹**

在 `seismo_skill/user_skills/<skill_name>/` 下创建 `SKILL.md`：

```text
seismo_skill/user_skills/my_skill/
├── SKILL.md
├── subskills/
│   └── station_metadata.md
├── references/
│   └── example_catalog.md
└── agents/
    └── debug_notes.md
```

`SKILL.md` 推荐写明：

```markdown
# 技能标题 / Skill Title

## 何时使用 / When to use

## 输入与输出 / Inputs and outputs

## 工作步骤 / Workflow

## 代码示例 / Examples
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
| 项目目录中的 `workflows/` | Web 项目工作流，推荐用于可复现研究和参数优化 |
| `~/.seismicx/workflows/` | 兼容旧版用户工作流；新项目建议优先保存在项目目录 |

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

将文件保存到当前项目的 `workflows/<name>.md`，使用上面展示的 frontmatter 格式。兼容模式仍可读取 `~/.seismicx/workflows/<name>.md`，但新项目建议保持项目内自包含。

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
| `save_user_workflow(name, text)` | 将 `.md` 工作流保存到项目目录；兼容旧版用户目录 |
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
│   ├── skills/                   # 内置 OpenAI-style 技能
│   │   └── pnsn_phase_detection/
│   │       ├── SKILL.md
│   │       └── pnsn/             # 技能内置 pnsn 代码和模型
│   ├── user_skills/              # Web 端生成/导入的用户技能
│   │   └── _gen_gmt_docs_zh/
│   │       ├── SKILL.md
│   │       └── subskills/
│   └── docs/                     # 可转换为 RAG / OpenAI-style SKILL 的文档源
│
├── conversational_agent.py       # 对话 Agent 核心（意图分类 + 技能执行）
├── config_manager.py             # LLM 配置管理
├── backend_manager.py            # 多后端支持（Ollama / vLLM / 在线 API）
├── seismic_cli.py                # 命令行入口
├── requirements.txt              # Python 依赖
└── logo.png

.sage_runtime/                    # 本地后台运行 PID、日志和环境信息（git ignored）
seismo_rag/                       # 项目知识库、索引和 project_config.json
```

---

## 配置文件

配置分为项目级和用户级两类。Web Config 页面会自动维护这些文件，通常不需要手动编辑。

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

常见位置：

- `seismo_rag/project_config.json`：项目级设置，例如搜索源、代码后端、技能/知识库辅助配置。
- `.sage_runtime/`：本地运行 PID、日志和临时环境信息，已加入 `.gitignore`。
- 科学分析/参数优化项目目录：保存项目输入、输出、图表、日志、Markdown/LaTeX 草稿和优化过程。
- `~/.seismicx/config.json`：少量用户级默认项，例如默认 LLM 后端。

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

前往 `/config` 选择一个已安装的 Ollama 模型，或配置在线 API 后点击「保存配置」。

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

在 `seismo_skill/user_skills/<skill_name>/` 下创建 OpenAI-style `SKILL.md`，按[创建自定义技能](#创建自定义技能)写明适用场景、输入输出、工作步骤和最小示例。保存后可在技能页刷新并管理，也会被 Chat、CodeEngine 和科学分析 Agent 检索使用。

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
