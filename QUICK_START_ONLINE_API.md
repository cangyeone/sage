# 🚀 快速开始：配置 Qwen/DeepSeek 等在线 API

## ⚡ 5 分钟快速设置

### 第 1 步：获取 API Key

选择一个服务商：

| 服务商 | 地址 | 推荐原因 |
|--------|------|---------|
| **DeepSeek** | https://platform.deepseek.com | 强大推理，性价比高 |
| **OpenAI** | https://platform.openai.com | 最强质量 |
| **SiliconFlow** (通义千问) | https://www.siliconflow.cn | 中文优化，有免费额度 |

### 第 2 步：配置系统（3 种方法选一）

#### 方法 A：Python 代码（推荐）
```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()

# 1. 选择服务商
cfg.set_llm_provider('deepseek')  # 或 'openai', 'siliconflow' 等

# 2. 输入 API Key
cfg.set_api_key('sk-xxxxxxxxxxxxx')

# 3. 自动检测并配置模型
models = cfg.get_online_api_models()
if models:
    cfg.set_llm_model(models[0])
    print(f"✓ 已配置: {models[0]}")
else:
    print("✗ 连接失败，请检查 API Key")
```

#### 方法 B：命令行工具
```bash
python demo_online_api_setup.py
# 按照提示操作
```

#### 方法 C：编辑配置文件
编辑 `~/.seismicx/config.json`：
```json
{
  "llm": {
    "provider": "deepseek",
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "sk-your-key-here",
    "model": "deepseek-v4-flash",
    "temperature": 0.3
  }
}
```

### 第 3 步：验证连接

```python
from conversational_agent import ConversationalAgent

agent = ConversationalAgent()
if agent.llm_agent:
    print("✓ 连接成功！可以开始对话了")
    result = agent.process_message("你好")
    print(result['response'])
else:
    print("✗ 连接失败:", agent.llm_error)
```

---

## 📋 配置示例

### DeepSeek 配置
```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()
cfg.set_llm_provider('deepseek')
cfg.set_api_key('sk-xxxxxxxxxxxxx')

# 检测可用模型
models = cfg.get_online_api_models()
# 输出: ['deepseek-v4-flash', 'deepseek-v4-pro']

cfg.set_llm_model('deepseek-v4-flash')
print("✓ DeepSeek 已配置")
```

### OpenAI 配置
```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()
cfg.set_llm_provider('openai')
cfg.set_api_key('sk-xxxxxxxxxxxxx')

# 检测可用模型
models = cfg.get_online_api_models()
cfg.set_llm_model('gpt-4o-mini')
print("✓ OpenAI 已配置")
```

### 通义千问配置（免费）
```python
from config_manager import LLMConfigManager

cfg = LLMConfigManager()
cfg.set_llm_provider('siliconflow')
cfg.set_api_key('sk-xxxxxxxxxxxxx')  # SiliconFlow API Key

# 检测可用模型
models = cfg.get_online_api_models()
# 输出: ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', ...]

cfg.set_llm_model('Qwen/Qwen2.5-7B-Instruct')
print("✓ 通义千问已配置")
```

---

## 🔧 故障排除

### 问题 1：API Key 被拒绝

```python
# 检查 API 配置
from config_manager import LLMConfigManager
cfg = LLMConfigManager()

# 验证 API Key
models = cfg.get_online_api_models()
if models is None:
    print("❌ API Key 可能不正确，请重新检查")
else:
    print(f"✓ API 连接正常，发现 {len(models)} 个模型")
```

### 问题 2：网络超时

```python
# 尝试指定更长的超时时间
from llm_agent import OpenAICompatibleClient

client = OpenAICompatibleClient(
    api_base='https://api.deepseek.com/v1',
    model='deepseek-v4-flash',
    api_key='sk-xxxxx',
    timeout=30  # 增加超时时间
)

if client.ping():
    print("✓ 连接成功")
```

### 问题 3：模型名称错误

```python
# 列出所有可用模型
from llm_agent import OpenAICompatibleClient

models = OpenAICompatibleClient.list_models(
    api_base='https://api.deepseek.com/v1',
    api_key='sk-xxxxx'
)

print("可用模型:")
for m in models:
    print(f"  - {m}")
```

---

## 💡 提示和技巧

### 1. 检测 API 可用模型

```bash
python test_online_api_models.py deepseek sk-xxxxx
```

### 2. 对比不同 API

| API | 速度 | 质量 | 成本 | 推荐 |
|-----|------|------|------|------|
| DeepSeek Flash | ⚡⚡⚡ | ⭐⭐⭐⭐ | ¥ | 平衡 |
| GPT-4o Mini | ⚡⚡ | ⭐⭐⭐⭐⭐ | ¥¥ | 质量优先 |
| Qwen2.5 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ¥ | 中文优化 |

### 3. 设置低温值以提高速度

```python
cfg = LLMConfigManager()
llm_cfg = cfg.get_llm_config()
llm_cfg['temperature'] = 0.1  # 默认 0.3，更低 = 更快更确定
cfg.save_config()
```

### 4. 使用 Flash/Mini 模型以降低成本

```python
cfg = LLMConfigManager()
cfg.set_llm_model('deepseek-v4-flash')  # 比 pro 便宜 90%
```

---

## 📚 进阶用法

### 直接使用客户端

```python
from llm_agent import OpenAICompatibleClient

client = OpenAICompatibleClient(
    api_base='https://api.deepseek.com/v1',
    model='deepseek-v4-flash',
    api_key='sk-xxxxx',
    temperature=0.3
)

# 测试连接
if client.ping():
    print("✓ 连接正常")

# 发送消息
response = client.chat(
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好"},
    ]
)
print(response['message']['content'])
```

### 创建自定义 Agent

```python
from conversational_agent import ConversationalAgent, SkillExecutor, ConversationContext
from llm_agent import OpenAICompatibleClient, OllamaToolAgent

# 创建客户端
client = OpenAICompatibleClient(
    api_base='https://api.deepseek.com/v1',
    model='deepseek-v4-flash',
    api_key='sk-xxxxx'
)

# 创建 Agent
executor = SkillExecutor()
context = ConversationContext()
agent = OllamaToolAgent(
    skill_executor=executor,
    context=context,
    client=client
)

# 使用 Agent
response = agent.process_message("分析一下地震数据")
print(response['response'])
```

---

## ✅ 完成检查清单

- [ ] 选择并注册了 API 服务商
- [ ] 获取了 API Key
- [ ] 运行配置脚本设置了 API
- [ ] 验证了模型列表可以获取
- [ ] 测试了与 Agent 的连接
- [ ] 进行了首次对话测试

---

## 🎓 下一步

1. **查看完整文档**: `ONLINE_API_SETUP.md`
2. **查看修改总结**: `ONLINE_API_MODIFICATION_SUMMARY.md`
3. **运行演示脚本**: `python demo_online_api_setup.py`
4. **测试工具**: `python test_online_api_models.py`

---

## 🆘 需要帮助？

查看以下文件了解更多：
- `ONLINE_API_SETUP.md` - 详细配置指南和常见问题
- `pnsn/LLM_SETUP_GUIDE.md` - LLM 设置完整指南
- 代码注释 - 详细的技术实现说明

**现在你可以使用任何 OpenAI-compatible API 来增强你的地震数据分析了！** 🎉
