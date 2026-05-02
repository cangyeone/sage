# 在线 API 模型配置指南

## 问题诊断

你的大模型接入在线 API 不工作的原因已找到并已修复！

### 原始问题
- **旧代码**只支持 Ollama 本地模型
- 当设置 Qwen/DeepSeek 的在线 API 时，系统会自动降级到规则引擎
- 没有自动检测在线 API 的模型列表功能

### 已修复项目
✅ 扩展了 `llm_agent.py`，添加 `OpenAICompatibleClient` 支持  
✅ 修改了 `build_agent_from_config()` 以支持在线 API  
✅ 添加了在线 API 模型列表自动检测功能  
✅ 更新了 `IntentClassifier` 以正确调用在线 API  
✅ 提供了模型列表获取工具

---

## 配置方法

### 方法 1：使用 Python API（推荐）

```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()

# 设置 DeepSeek
cfg.set_llm_provider('deepseek')
cfg.set_api_base('https://api.deepseek.com/v1')
cfg.set_api_key('your-deepseek-api-key-here')

# 自动检测可用模型
models = cfg.get_online_api_models()
if models:
    print(f"可用模型: {models}")
    # 选择第一个模型
    cfg.set_llm_model(models[0])
```

### 方法 2：直接编辑配置文件

配置文件位置：`~/.seismicx/config.json`

```json
{
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "your-api-key-here",
    "temperature": 0.3,
    "max_tokens": 2000
  }
}
```

---

## 支持的在线 API 服务商

### 🔷 DeepSeek
- **官网**: https://www.deepseek.com/
- **API 文档**: https://platform.deepseek.com/api-docs/
- **API Base**: `https://api.deepseek.com/v1`
- **推荐模型**:
  - `deepseek-v4-flash` (快速，适合实时应用)
  - `deepseek-v4-pro` (强大的推理)

### 🟢 OpenAI
- **官网**: https://openai.com/
- **API 文档**: https://platform.openai.com/docs/
- **API Base**: `https://api.openai.com/v1`
- **推荐模型**:
  - `gpt-4o-mini` (经济，性价比高)
  - `gpt-4o` (最强)

### 🔶 通义千问（SiliconFlow 代理）
- **官网**: https://www.siliconflow.cn/
- **API 文档**: https://docs.siliconflow.cn/
- **API Base**: `https://api.siliconflow.cn/v1`
- **推荐模型**:
  - `Qwen/Qwen2.5-7B-Instruct` (中文优化)
  - `Qwen/Qwen2.5-14B-Instruct` (更强)

### 🌙 Moonshot / Kimi
- **官网**: https://www.moonshot.cn/
- **API 文档**: https://platform.moonshot.cn/docs/
- **API Base**: `https://api.moonshot.cn/v1`
- **推荐模型**:
  - `moonshot-v1-8k`

### 🌩️ 阿里云 DashScope
- **官网**: https://dashscope.aliyun.com/
- **API Base**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **推荐模型**:
  - `qwen-turbo` (快速)
  - `qwen-plus` (均衡)

### 💠 智谱 AI
- **官网**: https://www.zhipuai.cn/
- **API Base**: `https://open.bigmodel.cn/api/paas/v4`
- **推荐模型**:
  - `glm-4-flash` (免费)
  - `glm-4` (强大)

---

## 快速开始

### 1️⃣ 获取 API Key

以 DeepSeek 为例：
1. 访问 https://platform.deepseek.com/
2. 创建账户并充值
3. 在 API Keys 页面创建新的 API Key
4. 复制 API Key

### 2️⃣ 检测可用模型

运行工具脚本：

```bash
# 检测当前配置
python test_online_api_models.py

# 检测特定 API
python test_online_api_models.py deepseek sk-xxxxxxxxxxxxxxxx
python test_online_api_models.py qwen sk-xxxxxxxxxxxxxxxx
```

### 3️⃣ 配置系统

```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()
cfg.set_llm_provider('deepseek')
cfg.set_api_key('sk-your-key-here')

# 自动获取并设置第一个可用模型
models = cfg.get_online_api_models()
if models:
    cfg.set_llm_model(models[0])
    print(f"✓ 已配置为: {models[0]}")
```

### 4️⃣ 测试连接

```python
from conversational_agent import ConversationalAgent

agent = ConversationalAgent()
if agent.llm_agent:
    print("✓ 在线 API 已连接，支持 AI 驱动的对话")
    result = agent.process_message("你好，我想检测地震数据")
    print(result['response'])
else:
    print("✗ 在线 API 连接失败，使用规则引擎模式")
```

---

## 常见问题

### Q: 为什么设置完 API 后还是不工作？

**A**: 请检查：
1. **API Key 是否正确**：在代码中测试 API Key
2. **网络连接**：确保能访问 API 端点
3. **模型名称**：确保模型名称与 API 返回的完全一致
4. **API Base URL**：检查是否包含 `/v1` 后缀（某些 API 需要）

### Q: 如何查看当前是否使用了在线 API？

**A**: 
```python
from conversational_agent import ConversationalAgent
agent = ConversationalAgent()
print(f"使用 LLM 路径: {agent.llm_agent is not None}")
print(f"错误信息: {agent.llm_error}")
```

### Q: 为什么 API 调用很慢？

**A**: 
- 在线 API 响应时间取决于网络和服务器
- 建议使用 `temperature=0.3` 以下的低温值以加快响应
- 某些模型比其他模型更快（如 `-flash` / `-mini` 版本）

### Q: 可以同时使用多个 API 吗？

**A**: 当前配置支持一个活跃 API。如果需要多个，可以：
1. 在不同的配置文件中切换
2. 在代码中动态创建多个 `OpenAICompatibleClient` 实例

---

## 深度集成

### 在自定义脚本中使用

```python
from llm_agent import OpenAICompatibleClient, OllamaToolAgent
from conversational_agent import SkillExecutor, ConversationContext
from config_manager import LLMConfigManager

# 获取配置
cfg = LLMConfigManager()
llm_cfg = cfg.get_llm_config()

# 创建 OpenAI-compatible 客户端
client = OpenAICompatibleClient(
    api_base=llm_cfg['api_base'],
    model=llm_cfg['model'],
    api_key=llm_cfg['api_key'],
)

# 创建 Agent
executor = SkillExecutor()
context = ConversationContext()
agent = OllamaToolAgent(
    skill_executor=executor,
    context=context,
    client=client,
)

# 使用 Agent
response = agent.process_message("检测数据目录下的所有地震波形")
print(response['response'])
```

### 获取在线 API 的模型列表

```python
from llm_agent import OpenAICompatibleClient

# 获取 DeepSeek 的所有模型
models = OpenAICompatibleClient.list_models(
    api_base='https://api.deepseek.com/v1',
    api_key='your-api-key'
)
print(f"可用模型: {models}")
```

---

## 技术细节

### 支持的 API 格式
系统支持 **OpenAI-compatible** API 格式，包括：
- `/v1/chat/completions` 端点
- `/v1/models` 或 `/models` 端点
- `Bearer {api_key}` 认证方式

### 响应格式兼容
自动处理的响应格式：
```json
{
  "data": [
    {"id": "model-name"},
    ...
  ]
}
```

或

```json
{
  "models": [
    {"name": "model-name"},
    ...
  ]
}
```

### 工具调用支持
在线 API 支持与 Ollama 相同的 tool-calling 格式，自动转换为 OpenAI 格式。

---

## 性能建议

| 场景 | 推荐模型 | API |
|------|---------|-----|
| **快速响应** | deepseek-v4-flash / gpt-4o-mini | DeepSeek / OpenAI |
| **中文优化** | Qwen2.5-7B-Instruct | SiliconFlow |
| **最高质量** | deepseek-v4-pro / gpt-4o | DeepSeek / OpenAI |
| **成本最低** | glm-4-flash | 智谱 AI |

---

## 更新日志

### v2025.04.28
✅ 添加 OpenAI-compatible API 支持  
✅ 自动模型列表检测  
✅ 多 provider 支持（DeepSeek、OpenAI、通义千问等）  
✅ 工具调用兼容性  

---

**需要帮助？** 查看 `pnsn/LLM_SETUP_GUIDE.md` 了解更多详情。
