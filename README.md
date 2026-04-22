# SeismicX

SeismicX 是一个面向地震学研究的 AI 辅助分析平台，提供基于对话的波形浏览、震相拾取、震相关联与极性分析功能。用户可以通过自然语言（中文/英文）向系统下达指令，无需记忆命令行参数。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 🗂 波形浏览 | 列出目录下的地震波形文件，支持 SAC / MSEED / SEED 等格式 |
| 📈 波形可视化 | 在对话窗口内直接渲染三分量交互式波形图（Plotly.js） |
| 🔍 震相拾取（单台） | 对指定台站的三分量数据进行在线 JIT 模型推理，检测 Pg/Sg/Pn/Sn/P/S 震相并叠加到波形图 |
| 📂 批量拾取（目录） | 遍历整个目录，自动识别数据格式、分组三分量、批量推理，实时显示进度 |
| 🔗 震相关联 | 将多台站拾取结果关联为地震事件（FastLink / REAL / Gamma） |
| 🧭 极性分析 | 分析 P 波初动方向 |
| ⚙️ 配置自动检测 | 自动识别文件扩展名、采样率、通道索引，更新 `pnsn/config/picker.py` |

---

## 系统要求

- Python ≥ 3.8
- PyTorch ≥ 1.9（CPU 即可，GPU 可选）
- ObsPy ≥ 1.3
- Flask ≥ 2.0

```bash
pip install torch obspy flask numpy scipy
```

---

## 安装与启动

```bash
git clone https://github.com/yourname/sage.git
cd sage

# 安装依赖
pip install torch obspy flask numpy scipy

# 启动 Web 服务（默认端口 5050）
python web_app/app.py

# 指定端口和主机
python web_app/app.py --port 8080 --host 0.0.0.0
```

浏览器访问 `http://localhost:5050`，进入对话界面。

---

## 目录结构

```
sage/
├── web_app/
│   ├── app.py                  # Flask 主服务
│   └── templates/
│       └── chat.html           # 对话界面（含 Plotly.js 波形渲染）
├── pnsn/
│   ├── sage_picker.py          # 批量拾取核心模块（SagePicker）
│   ├── picker.py               # 传统命令行批量拾取器
│   ├── pickers/
│   │   └── pnsn.v3.jit         # 预训练 JIT 震相拾取模型
│   ├── config/
│   │   └── picker.py           # 数据参数配置（自动更新）
│   └── models/                 # 模型定义（UNet、EQTransformer 等）
├── conversational_agent.py     # 意图分类 + 技能路由
├── llm_agent.py                # LLM 对话后端（可选）
├── config_manager.py           # LLM 提供商配置
└── results/                    # 拾取输出文件（自动创建）
```

---

## 使用说明

### 对话式操作

在浏览器对话框中直接输入自然语言，例如：

```
拾取一下 /data/waveforms 目录下所有的震相
绘制第 3 个文件的波形
拾取 X1.53085 台站的震相
遍历 /data 目录批量拾取震相
查看当前目录下的 SAC 文件
```

系统会自动识别意图，调用对应功能。

---

### 批量震相拾取

对话中提及目录路径，系统将：

1. **预扫描**：用 ObsPy 读取目录中所有可识别文件，按 `NET.STA.LOC` 分组
2. **询问处理方式**（若存在不足三分量台站）：
   - **跳过**：只处理有完整三分量的台站
   - **复制**：将现有分量复制补齐，仍可拾取（精度略低）
3. **后台拾取**：在后台线程中执行，对话界面每 3 秒刷新一次进度

进度显示示例：

```
⚙️ 12 / 47 台站 | 已检出 89 个震相
当前: X1.53085.
```

完成后显示：

```
✅ 批量拾取完成！
处理台站: 47 | 检出震相: 312
输出文件: results/sage_picks_20260422_194512.txt
```

#### 输出文件格式

```
##数据格式为:
##数据位置
##震相,相对时间（秒）,置信度,绝对时间（%Y-%m-%d %H:%M:%S.%f）,信噪比,振幅均值,台站,初动,初动概率
#/data/waveforms/X1.53085.
Pg,12.340,0.923,2022-05-21 07:26:47.340000,8.521,0.000142,X1.53085.,N,0.000
Sg,22.150,0.871,2022-05-21 07:26:57.150000,6.103,0.000089,X1.53085.,N,0.000
```

---

### 命令行批量拾取（不使用 Web 界面）

```bash
cd sage
python pnsn/sage_picker.py \
    -i /path/to/data \
    -o results/my_picks \
    -m pnsn/pickers/pnsn.v3.jit \
    --incomplete skip        # 或 duplicate
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i` | 必填 | 输入数据目录 |
| `-o` | 必填 | 输出文件前缀（自动添加 .txt / .log / .err） |
| `-m` | 必填 | JIT 模型路径 |
| `-r` / `--samplerate` | `100` | 目标采样率（Hz） |
| `-d` / `--device` | `cpu` | 推理设备（`cpu` 或 `cuda`） |
| `--incomplete` | `skip` | 不足三分量处理：`skip` 跳过 / `duplicate` 复制补齐 |

---

### Python API

```python
from pnsn.sage_picker import SagePicker

picker = SagePicker(
    model_path='pnsn/pickers/pnsn.v3.jit',
    samplerate=100.0,
    device='cpu'
)

# 预扫描（不加载波形，速度快）
info = picker.scan_directory('/path/to/data')
print(f"完整台站: {info['n_complete']}，不足三分量: {info['n_incomplete']}")

# 批量拾取
result = picker.pick_directory(
    input_dir='/path/to/data',
    output_base='results/picks',
    incomplete='skip',            # 'skip' | 'duplicate'
    progress_cb=lambda sta, done, total, n_picks:
        print(f"{done}/{total} {sta}  picks={n_picks}")
)
print(f"台站: {result['n_stations']}  震相: {result['n_picks']}")
print(f"输出: {result['output']}")
```

---

## 模型说明

### 默认模型：pnsn.v3.jit

- 格式：PyTorch TorchScript（`.jit`）
- 输入：`[N, 3]` float32 张量，列顺序为 E / N / Z
- 输出：`[n_picks, 3]` 张量，每行为 `[phase_type, sample_index, confidence]`
- 采样率：100 Hz
- 震相类型编码：

| 编码 | 震相 |
|------|------|
| 0 | Pg |
| 1 | Sg |
| 2 | Pn |
| 3 | Sn |
| 4 | P |
| 5 | S |

### 更换模型

在对话中指定模型名称，或修改 `pnsn/config/picker.py`。系统每次批量拾取前会自动重新检测数据格式并更新配置文件。

---

## 支持的数据格式

系统使用 ObsPy 读取波形，凡是 ObsPy 支持的格式均可使用，包括但不限于：

- **SAC**（`.sac`）
- **MiniSEED**（`.mseed`、`.miniseed`、`.MSEED`）
- **SEED**（`.seed`）

文件命名无强制要求，系统通过 ObsPy 直接解析文件头获取网络、台站、通道等元数据。推荐文件名格式：

```
NET.STA.LOC.CHANNEL.D.YYYYMMDDHHMMSS.sac
例：X1.53085.01.BHZ.D.20220521072623.sac
```

---

## 配置文件

`pnsn/config/picker.py` 在每次批量拾取前由系统自动更新，也可手动修改：

```python
class Parameter:
    nchannel   = 3              # 通道数
    samplerate = 100            # 采样率（Hz）
    prob       = 0.3            # 拾取概率阈值
    nmslen     = 1000           # NMS 窗口长度（采样点）
    filenametag = ".sac"        # 文件扩展名
    namekeyindex = [0, 1]       # NET/STA 在文件名中的字段索引
    channelindex = 3            # 通道名在文件名中的字段索引
    chnames    = [['BHE','BHN','BHZ']]   # 有效通道组合
    bandpass   = [1, 10]        # 信噪比计算带通滤波范围（Hz）
```

---

## 常见问题

**Q：对话时说"我需要更多信息"？**  
A：确保消息中包含完整的目录路径（以 `/` 开头），例如 `拾取 /data/ev001 目录下所有震相`。

**Q：拾取进度条一直不动？**  
A：数据量大时，系统首先需要遍历并读取所有文件头（预扫描阶段），完成后进度条才开始更新。如目录中有大量非地震数据文件，此步骤可能略慢。

**Q：批量拾取结果为 0 个震相？**  
A：检查同路径下的 `.err` 文件，其中记录了每个台站失败的原因。常见原因：三分量时间不重叠、数据段过短、重采样后长度不足。

**Q：如何使用 GPU 加速？**  
A：安装 CUDA 版 PyTorch 后，在命令行参数中添加 `-d cuda`，或在 `SagePicker` 初始化时指定 `device='cuda'`。

---

## 开发说明

### 新增模型

1. 将模型转换为 TorchScript：参考 `pnsn/makejit.*.py` 系列脚本
2. 将 `.jit` 文件放入 `pnsn/pickers/`
3. 在对话中指定模型名称即可调用

### 修改意图识别

编辑 `conversational_agent.py` 中 `IntentClassifier.intent_patterns`，添加新的关键词和正则模式。

### Web API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 对话接口，接收 `{"message": "..."}` |
| `/api/task/<task_id>` | GET | 查询后台任务状态与实时进度 |
| `/api/chat_picks/<task_id>` | GET | 查询单台拾取结果（含解析后的 picks 列表） |
| `/api/tasks` | GET | 列出所有任务 |

---

## 许可证

MIT License
