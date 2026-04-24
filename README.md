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
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
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
  - [pnsn 震相拾取模块](#pnsn-震相拾取模块安装)
  - [RAG 功能依赖](#rag-功能依赖)
- [配置 LLM 后端](#配置-llm-后端)
- [Web 界面](#web-界面)
- [命令行工具](#命令行工具)
- [对话路由机制](#对话路由机制)
- [seismo_skill 技能系统](#seismo_skill-技能系统)
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
| 🧑‍💻 **代码生成执行** | LLM 生成 Python 代码 + 沙箱安全执行 + 内置地震学工具包，串联多个技能步骤 |
| 🗺️ **GMT 地图绘制** | 调用 GMT6 绘制震中图、台站图、地形图、震源机制球，图像与脚本均可下载 |
| 🤖 **自主 Agent** | 读入论文 → 理解方法 → 自主规划 → 逐步编程实现，每步自动重试 |
| 📚 **知识库 RAG** | BGE-M3 向量化 + FAISS 检索，持久化存储，批量 PDF 入库与文献问答 |
| 📖 **文献解读** | 临时上传 PDF → 深度解读方法/公式/结论，多轮追问 |
| 🗂 **本地文件访问** | 授权指定目录后，LLM 可直接读取文件列表辅助分析 |
| ⚡ **技能系统** | Markdown 格式技能文档（7 个内置 + 无限自定义），对话和代码生成时自动检索注入 |
| 📈 **波形可视化** | 对话窗口内嵌波形图（震相标注叠加），图像可点击放大或下载 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI (Flask + JS)                          │
│        /chat  ·  /knowledge  ·  /skills  ·  /llm-settings       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST API
┌──────────────────────────▼──────────────────────────────────────┐
│                      /api/chat/route                             │
│                   LLM 意图分类路由器                              │
│         code ──────────┬────────── qa ──────── chat             │
└─────────┬──────────────┼──────────────┬────────────────────────┘
          │              │              │
  ┌───────▼──────┐  ┌────▼─────┐  ┌────▼──────┐
  │ CodeEngine   │  │ RAG 问答  │  │ 通用对话  │
  │ + Toolkit    │  │ BGE-M3   │  │           │
  │ + GMT        │  │ + FAISS  │  │           │
  └───────┬──────┘  └──────────┘  └───────────┘
          │
  ┌───────▼──────────────────────────────────────┐
  │            seismo_skill 技能检索              │
  │    内置 7 个技能  +  用户自定义技能            │
  │    (~/.seismicx/skills/)                     │
  └───────┬──────────────────────────────────────┘
          │ 自动注入函数说明 + 代码示例
  ┌───────▼──────────────────────────────────────┐
  │            LLM Backend                       │
  │   Ollama (本地)  ·  vLLM  ·  OpenAI 兼容     │
  └──────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │           pnsn/ 震相拾取引擎                 │
  │    PhaseNet / EQTransformer / JIT / ONNX    │
  │    FastLink / Gamma 震相关联                 │
  └─────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 1. 克隆主仓库
git clone https://github.com/yourname/sage.git
cd sage

# 2. 克隆 pnsn 震相拾取模块（必须放在 sage/ 目录下）
git clone https://github.com/cangyeone/pnsn.git

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Ollama 并拉取模型（选一个）
ollama serve &
ollama pull qwen3:8b          # 轻量，约 6 GB

# 5. 启动 Web 服务
python web_app/app.py --port 5010

# 6. 浏览器访问
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
git clone https://github.com/yourname/sage.git
cd sage

# 完整安装（推荐）
pip install -r requirements.txt

# 或按需安装各部分
pip install flask flask-cors                          # Web 服务
pip install obspy torch scipy numpy pandas            # 地震数据处理
pip install matplotlib plotly                         # 可视化
pip install FlagEmbedding faiss-cpu pdfminer.six PyMuPDF  # RAG 知识库
```

### pnsn 震相拾取模块安装

pnsn 是专门用于震相拾取的深度学习模型库，由 [cangyeone](https://github.com/cangyeone) 开发。**必须将其克隆到 `sage/` 主目录下**，SAGE 通过相对路径调用。

```bash
# 在 sage/ 目录下执行
git clone https://github.com/cangyeone/pnsn.git

# 安装 pnsn 自身依赖
cd pnsn
pip install -r requirements.txt
cd ..
```

**目录结构确认：**

```
sage/
├── pnsn/               ← 必须在此位置
│   ├── sage_picker.py
│   ├── fastlinker.py
│   ├── gammalink.py
│   ├── pickers/        ← JIT / ONNX 模型文件
│   └── config/
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

无需重启即可扩展 AI 能力。

- 左侧：内置技能（只读）与用户自定义技能（可编辑/删除）分组展示
- 右侧：Markdown 编辑器 + 实时预览，含语法高亮
- 支持新建、编辑、删除自定义技能
- 保存后下一次对话或代码生成立即生效

> 自定义技能存储路径：`~/.seismicx/skills/`

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
    -m pnsn/pickers/pnsn.v3.jit

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

### `seismo_code/` — 代码生成与执行引擎

```
seismo_code/
├── code_engine.py      # LLM 代码生成（含技能注入、多轮历史、错误重试）
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
│   └── gmt_plotting.md           # GMT 地图绘制
│
├── seismo_code/                  # 代码生成与执行引擎
│   ├── code_engine.py            # LLM 代码生成（多轮历史 + 错误重试）
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
├── pnsn/                         # ← 需单独 clone（见安装说明）
│   ├── sage_picker.py            # 批量拾取主类（SagePicker）
│   ├── fastlinker.py             # FastLink 震相关联
│   ├── gammalink.py              # Gamma 震相关联
│   ├── pickers/                  # JIT / ONNX 模型文件
│   └── config/                   # 拾取器参数配置
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
└── skills/                       # 用户自定义技能文档
    └── my_custom_skill.md
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
