# SeismicX LLM Settings & Connection Architecture — Exploration Report

**Generated**: 2026-05-01  
**Codebase**: /Users/yuziye/Documents/GitHub/sage  
**Status**: ✅ Thoroughly Explored

---

## Executive Summary

The SeismicX application has a **multi-backend LLM architecture** supporting:
- **Ollama** (local models)
- **Online APIs** (DeepSeek, OpenAI, Qwen/SiliconFlow, Moonshot, DashScope/Alibaba, Zhipu, Anthropic, custom)

A recent modification (April 28, 2026) fixed critical issues where **online LLM APIs were not functioning**. The fixes include:
1. Added `OpenAICompatibleClient` class for OpenAI-format APIs
2. Extended `build_agent_from_config()` to support 8+ online providers
3. Implemented automatic model list detection from APIs
4. Fixed Authorization header handling in API calls
5. Created web UI with preset buttons for quick API configuration

**Key Issue Identified**: Model names are currently **hardcoded** in the UI and **not fetched from APIs** by default. This requires manual input or API calls to detect available models.

---

## 1. Directory Structure & Key Files

### Core LLM Configuration Files
```
/sessions/compassionate-dazzling-mccarthy/mnt/sage/
├── config_manager.py                    (Main config mgmt, model detection)
├── llm_agent.py                         (LLM client implementations)
├── backend_manager.py                   (Unified backend detection)
├── conversational_agent.py              (Integration layer)
│
├── test_online_api_models.py            (Model list detection tool)
├── demo_online_api_setup.py             (Interactive setup wizard)
│
├── web_app/
│   ├── app.py                          (Flask entry point)
│   ├── state.py                        (Shared state)
│   ├── routes/
│   │   ├── llm.py                      (API endpoints for LLM settings)
│   │   ├── chat.py                     (Chat endpoints using LLM)
│   │   └── ...
│   └── templates/
│       └── llm_settings.html           (Web UI for LLM config)
│
├── Documentation/
│   ├── ONLINE_API_SETUP.md             (Configuration guide)
│   ├── ONLINE_API_MODIFICATION_SUMMARY.md (Technical summary)
│   ├── QUICK_START_ONLINE_API.md       (Quick reference)
│   └── README.md                        (Project overview)
```

---

## 2. LLM Model Settings Page (Web UI)

**File**: `/sessions/compassionate-dazzling-mccarthy/mnt/sage/web_app/templates/llm_settings.html`

### Features Implemented
✅ **Provider Selection Cards**:
- Ollama (Local)
- OpenAI
- DeepSeek
- Custom API (SiliconFlow, Moonshot, Alibaba, Zhipu, etc.)

✅ **Ollama-Specific UI**:
- Displays installed models (fetched from `ollama list`)
- Custom model input field
- Model download section with recommended chips
- Download progress tracking via `/api/llm/ollama/pull` and `/api/llm/ollama/pull/status`

✅ **Online API UI**:
- DeepSeek section: Shows preset buttons for `deepseek-v4-flash` and `deepseek-v4-pro`
- Generic online section: API Base URL, Model Name, API Key inputs
- Quick fill presets: OpenAI, SiliconFlow, Moonshot, DashScope (Alibaba), Zhipu

✅ **Current Model Badge**: Displays active model with provider name

✅ **Test Connection**: `/api/llm/config/test` endpoint to validate configuration

### HTML Form IDs
```javascript
provider              // Hidden input: 'ollama' | 'deepseek' | 'openai' | 'custom'
ollamaCustomModel     // Text: Manual model name for Ollama
onlineModel           // Text: Model for DeepSeek (deepseek-v4-flash, etc.)
onlineModelGeneric    // Text: Model for OpenAI/custom (gpt-4o, etc.)
apiBase              // Text: https://api.openai.com/v1
apiKey               // Password: sk-...
downloadModelInput   // Text: Model to download via ollama pull
```

---

## 3. Backend LLM Routes & API Endpoints

**File**: `/sessions/compassionate-dazzling-mccarthy/mnt/sage/web_app/routes/llm.py`

### Key Endpoints

#### GET `/api/llm/config`
- Returns: `{ config, first_run, ollama_available }`
- Reads from `~/.seismicx/config.json`
- **Issue**: API key is masked in response for security

#### POST `/api/llm/config`
- Accepts: `{ provider, model, api_base, api_key, temperature, max_tokens }`
- Calls `LLMConfigManager` to persist settings
- **Limitation**: Does NOT automatically fetch available models

#### GET `/api/llm/ollama/models`
- Returns: `{ installed: [...], recommended: [...], ollama_available: bool }`
- Fetches from `ollama list` command
- **Status**: ✅ Working

#### POST `/api/llm/ollama/pull`
- Pulls a model using streaming API
- **Implementation**: Background thread + polling for status
- **Status**: ✅ Working

#### GET `/api/llm/ollama/pull/status`
- Returns progress: `{ status, progress, detail, error }`
- **Status**: ✅ Working

### Missing Endpoints
❌ **No GET `/api/llm/online/models`**: Cannot fetch model list from online APIs via REST endpoint
- Current workaround: Must use Python `LLMConfigManager.get_online_api_models()`

---

## 4. Configuration Manager & Model Detection

**File**: `/sessions/compassionate-dazzling-mccarthy/mnt/sage/config_manager.py`

### Key Methods

#### `get_llm_config()`
```python
{
    'provider': 'deepseek' | 'openai' | 'ollama' | 'custom' | ...,
    'model': 'model-name',
    'api_base': 'https://api.../v1',
    'api_key': 'sk-...',
    'temperature': 0.3,
    'max_tokens': 2000
}
```

#### `get_online_api_models(provider=None)` ⭐ **KEY METHOD**
```python
models = cfg.get_online_api_models()
# Returns: ['deepseek-v4-flash', 'deepseek-v4-pro', ...]
# Or: None if API unreachable or API key missing
```

**How it works**:
1. Reads provider from config (or parameter)
2. Gets API base from config OR hardcoded presets
3. Calls `/v1/models` or `/models` endpoint with Bearer auth
4. Parses responses in multiple formats:
   - OpenAI format: `{ "data": [{"id": "model-name"}] }`
   - Ollama format: `{ "models": [{"name": "model-name"}] }`
5. Returns list or None

**Supported Providers**:
- ✅ OpenAI
- ✅ DeepSeek
- ✅ SiliconFlow (Qwen)
- ✅ Moonshot
- ✅ DashScope (Alibaba)
- ✅ Zhipu
- ✅ Anthropic

### Provider Presets (Hardcoded)
```python
PRESETS = {
    'openai': {
        'base': 'https://api.openai.com/v1',
        'model': 'gpt-4o'
    },
    'siliconflow': {
        'base': 'https://api.siliconflow.cn/v1',
        'model': 'Qwen/Qwen2.5-72B-Instruct'
    },
    'moonshot': {
        'base': 'https://api.moonshot.cn/v1',
        'model': 'moonshot-v1-8k'
    },
    'dashscope': {
        'base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-turbo'
    },
    # ... more providers
}
```

**Issue**: DashScope/Alibaba preset uses `/compatible-mode/v1` endpoint (different from standard `/v1`)

---

## 5. LLM Agent Architecture

**File**: `/sessions/compassionate-dazzling-mccarthy/mnt/sage/llm_agent.py`

### Client Implementations

#### `OllamaClient` (Original)
```python
class OllamaClient:
    def __init__(self, api_base: str, model: str, temperature: float)
    def chat(messages, tools=None) -> Dict
    def ping() -> bool
```
- Calls `/api/chat` endpoint with streaming
- Supports tool calling via `tools` parameter
- **Status**: ✅ Working

#### `OpenAICompatibleClient` (New - Apr 28, 2026)
```python
class OpenAICompatibleClient:
    def __init__(self, api_base: str, model: str, api_key: str, ...)
    def chat(messages, tools=None) -> Dict
    def ping() -> bool
    @staticmethod
    def list_models(api_base, api_key) -> Optional[List[str]]
```
- Calls `/v1/chat/completions` endpoint
- Converts OpenAI tool-calls to Ollama format
- **Status**: ✅ Working
- **Supports**: DeepSeek, OpenAI, SiliconFlow, Moonshot, DashScope, Zhipu, Anthropic, custom

### Agent Assembly: `build_agent_from_config()`

**Location**: `llm_agent.py:631`

```python
def build_agent_from_config(skill_executor, context):
    cfg = get_config_manager().get_llm_config()
    provider = cfg.get("provider")
    
    # Branch 1: Ollama
    if provider == "ollama":
        client = OllamaClient(...)
        agent = OllamaToolAgent(skill_executor, context, client=client)
        if not client.ping():
            return None
        return agent
    
    # Branch 2: Online APIs
    elif provider in ["deepseek", "openai", "siliconflow", ...]:
        client = OpenAICompatibleClient(...)
        agent = OllamaToolAgent(skill_executor, context, client=client)
        if not client.ping():
            return None
        return agent
    
    return None  # Unsupported provider
```

### Integration: `OllamaToolAgent`
**Location**: `llm_agent.py:505`

```python
class OllamaToolAgent:
    def __init__(self, skill_executor, context, client=None, ...):
        if client is None:
            self.client = OllamaClient(...)  # backward compatible
        else:
            self.client = client  # accepts both OllamaClient & OpenAICompatibleClient
    
    def process_message(user_msg) -> Dict:
        # Works with either client transparently
        response = self.client.chat(messages, tools)
        # Parse, execute tools, format response
        return {...}
```

**Key Feature**: Abstracted client interface allows both local and online backends to work transparently.

---

## 6. Conversational Agent Integration

**File**: `/sessions/compassionate-dazzling-mccarthy/mnt/sage/conversational_agent.py`

### LLM Path vs Rule-Based Fallback

```python
class ConversationalAgent:
    def __init__(self):
        self.config_manager = get_config_manager()
        try:
            self.llm_agent = build_agent_from_config(...)
        except Exception as e:
            self.llm_error = str(e)
            self.llm_agent = None  # Falls back to rule-based
    
    def process_message(user_msg) -> Dict:
        # ---- LLM Path (Preferred) ----
        if self.llm_agent is not None:
            try:
                result = self.llm_agent.process_message(user_msg)
                return result
            except Exception as e:
                self.llm_error = str(e)
                self.llm_agent = None
                # Fall through to rule-based
        
        # ---- Rule-Based Fallback ----
        intent = self.intent_classifier.classify(user_msg)
        # ... execute skills based on intent
```

**Graceful Degradation**: If online API is misconfigured or unreachable, system automatically reverts to rule-based pattern matching.

---

## 7. Online API Connection Testing

### Python Testing Tools

#### `test_online_api_models.py`
```bash
# Test current config
python test_online_api_models.py

# Test specific provider
python test_online_api_models.py deepseek sk-xxxxx
python test_online_api_models.py qwen sk-xxxxx
```

**Implementation**:
```python
def test_provider(provider: str, api_key: str):
    api_base = ONLINE_PROVIDERS[provider]["api_base"]
    models = OpenAICompatibleClient.list_models(api_base, api_key)
    if models:
        print(f"✓ {len(models)} models found")
```

#### `demo_online_api_setup.py`
Interactive wizard with 3 demos:
1. Configure DeepSeek API + detect models
2. Test LLM Agent connection
3. Show all supported providers

---

## 8. Current Implementation Issues & Root Causes

### Issue #1: Online APIs Not Connecting Effectively ❌

**Root Causes Identified**:

1. **Missing Web API Endpoint for Model List**
   - Frontend cannot call `/api/llm/online/models` to fetch available models
   - Users must manually type model names or use Python CLI
   - **Location**: Not in `web_app/routes/llm.py`
   - **Fix Required**: Add endpoint similar to `/api/llm/ollama/models`

2. **Model Names Hardcoded in UI**
   - DeepSeek section shows only 2 preset models (flash, pro)
   - Generic section has placeholder "gpt-4o / Qwen2.5-72B-Instruct / …"
   - Users cannot see full model list without Python
   - **File**: `llm_settings.html` (lines with preset buttons)
   - **Fix Required**: Fetch and populate model dropdown dynamically

3. **API Key Validation Issue**
   - Form cannot validate if API key is correct before saving
   - User must save → test → realize it's wrong → re-enter
   - **Location**: `llm.py:update_llm_config()` has no validation endpoint
   - **Fix Required**: Add `/api/llm/test-connection` endpoint

4. **DashScope/Alibaba Endpoint Difference**
   - Preset uses `/compatible-mode/v1` but model list fetching tries `/v1/models`
   - May cause connection failures
   - **File**: `config_manager.py:96`
   - **Issue**: Inconsistent endpoint naming

5. **Missing Error Messages on UI**
   - Test Connection button shows "Connection failed: ..." but no details
   - User doesn't know if it's auth, network, or invalid model
   - **Location**: Frontend toast notification
   - **Fix Required**: Return more detailed error messages from backend

### Issue #2: Model Names Not Auto-Fetched ❌

**Problem**: Users must:
1. Know the exact model name beforehand
2. Or use Python CLI: `python test_online_api_models.py deepseek sk-xxx`

**Missing Features**:
- No "Refresh Models" button on web UI
- No dropdown showing available models for the configured API
- No model name validation before saving

**How It Should Work** (Proposed):
```
User → Select Provider (deepseek) → Enter API Key → 
  [Fetch Models Button] → Dropdown with [deepseek-v4-flash, deepseek-v4-pro] → 
  Select Model → Save
```

**Current** (Broken):
```
User → Select Provider (deepseek) → Enter API Key → 
  Type model name manually (deepseek-v4-flash) → Save → 
  Test Connection → Fails (because they guessed wrong)
```

### Issue #3: Alibaba/DashScope API Not Working ❌

**Symptoms**: DashScope API configuration fails even with correct API key

**Root Cause Analysis**:
1. DashScope uses different endpoint: `/compatible-mode/v1/chat/completions`
2. Model list endpoint might differ: `/compatible-mode/v1/models`
3. Authorization header format might differ
4. Preset in `config_manager.py` is correct but `get_online_api_models()` tries generic endpoints

**File Issues**:
- `config_manager.py:108-115`: Generic `/v1/models` endpoint fetching
- `llm_agent.py:461`: Generic `OpenAICompatibleClient.ping()` tries `/models` or `/v1/models`

**What Works**:
- ✅ Preset URL is correct: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- ✅ API key format is correct: `Bearer sk-...`
- ✅ Chat API endpoint should work: `/chat/completions`

**What Doesn't Work**:
- ❌ Model list detection tries wrong endpoints

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Web UI: llm_settings.html                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Provider Selection (Ollama / OpenAI / DeepSeek...)   │   │
│  │ API Key Input | Model Name Input | Test Connection  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/llm/config
                       │ GET  /api/llm/config
                       │ POST /api/llm/ollama/pull
                       │ GET  /api/llm/ollama/models
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│  Backend Routes: web_app/routes/llm.py                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ - llm_config_get()       → read ~/.seismicx/config   │   │
│  │ - update_llm_config()    → write config + validate   │   │
│  │ - get_ollama_models()    → call `ollama list`        │   │
│  │ - pull_ollama_model()    → thread + streaming        │   │
│  │ - pull_ollama_status()   → poll progress             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│  Config Manager: config_manager.py                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ get_llm_config()              → read from JSON       │   │
│  │ set_llm_provider/model/key()  → write to JSON        │   │
│  │ get_online_api_models()       → fetch from API       │   │
│  │ check_ollama_available()      → test ollama CLI      │   │
│  │ get_ollama_models()           → parse `ollama list`  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            v                     v
   ┌────────────────┐    ┌──────────────────┐
   │  Conversational│    │ Conversational   │
   │  Agent         │    │ Agent            │
   │                │    │ (LLM Path)       │
   └────────┬───────┘    └────────┬─────────┘
            │                     │
            │                     v
            │            ┌──────────────────┐
            │            │ llm_agent.py     │
            │            │                  │
            │            │ - OllamaClient   │
            │            │ - OpenAI-compat  │
            │            │ - OllamaToolAgent│
            │            └────────┬─────────┘
            │                     │
            │            ┌────────┴──────────┐
            │            │                   │
            │            v                   v
            │        [Ollama API]    [DeepSeek/OpenAI/...]
            │     localhost:11434    api.deepseek.com/v1
            │                        api.openai.com/v1
            │                        dashscope.aliyuncs.com/...
            │
            └────────────────────────────────────┘
                  (Falls back if LLM unavailable)
```

---

## 10. Summary of Key Code Locations

| Component | File | Line(s) | Status |
|-----------|------|---------|--------|
| **Web UI** | `web_app/templates/llm_settings.html` | All | ✅ Deployed |
| **Backend Routes** | `web_app/routes/llm.py` | All | ✅ Deployed |
| **Config Manager** | `config_manager.py` | All | ✅ Deployed |
| **LLM Clients** | `llm_agent.py:207-355` (Ollama) | 207-355 | ✅ Working |
| **LLM Clients** | `llm_agent.py:396-489` (OpenAI-compat) | 396-489 | ✅ Working (new) |
| **Agent Assembly** | `llm_agent.py:631-712` | 631-712 | ✅ Working (new) |
| **OllamaToolAgent** | `llm_agent.py:505-630` | 505-630 | ✅ Working |
| **ConversationalAgent** | `conversational_agent.py:2181-2270` | 2181-2270 | ✅ Working |
| **Model Detection Tool** | `test_online_api_models.py` | All | ✅ Available |
| **Setup Wizard** | `demo_online_api_setup.py` | All | ✅ Available |

---

## 11. Alibaba/DashScope Configuration Details

### Current State
- ✅ **API Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1` (Correct)
- ✅ **Chat Endpoint**: `/chat/completions` (Standard OpenAI format)
- ✅ **API Key Format**: `sk-...` with Bearer auth
- ✅ **Preset Configured**: Yes, in `config_manager.py:PRESETS['dashscope']`
- ✅ **UI Support**: Yes, quick-fill button "阿里通义" in HTML

### Known Issues
1. **Model List Endpoint**: `/v1/models` may not be available on DashScope
   - Try: `/compatible-mode/v1/models`
   - Current code tries generic `/v1/models` first
   - **File**: `config_manager.py:108-115`

2. **Response Format**: Model list might use different structure
   - Standard OpenAI: `{"data": [{"id": "model-name"}]}`
   - DashScope: Possibly `{"models": [...]}`
   - Current code supports both formats

3. **Default Model**: `qwen-turbo` is correct default

### Testing DashScope
```bash
python test_online_api_models.py dashscope sk-xxxxx
```

If it fails:
1. Check API key validity
2. Try chat endpoint directly: `/compatible-mode/v1/chat/completions`
3. DashScope may not expose `/models` endpoint (unlike OpenAI)

---

## 12. Recommended Fixes Priority

### P0 (Critical - Blocks Online API Usage)
1. **Add `/api/llm/online/models` endpoint**
   - Fetches model list for configured provider
   - Called when user selects online provider
   - Returns: `{ "models": [...], "error": "..." }`
   - **Effort**: 20 lines in `llm.py`

2. **Add "Fetch Models" button to Web UI**
   - Calls above endpoint
   - Populates model dropdown
   - Shows loading state
   - **Effort**: 50 lines in `llm_settings.html`

3. **Fix DashScope Model List**
   - Test if `/compatible-mode/v1/models` works
   - Update fallback endpoint list in `config_manager.py`
   - **Effort**: 5 lines change

### P1 (High - UX Improvement)
4. **Add API Key Validation Endpoint**
   - Test connection before saving
   - Return: `{ "valid": bool, "message": "...", "models": [...] }`
   - **Effort**: 30 lines in `llm.py`

5. **Improve Error Messages**
   - Distinguish: auth error, network error, invalid model, timeout
   - Show in test connection response
   - **Effort**: 50 lines in `llm.py` + frontend

6. **Add Model Dropdown to Web UI**
   - Auto-populate from API
   - Show model capabilities (if available)
   - **Effort**: 100 lines in HTML + JS

### P2 (Nice-to-Have)
7. **Cache Model Lists**
   - Avoid re-fetching on every page load
   - Cache for 1 hour
   - **Effort**: 30 lines in `config_manager.py`

8. **Support Multiple APIs**
   - Allow configuring backup APIs
   - Automatic fallback if primary fails
   - **Effort**: 100+ lines

---

## 13. Configuration File Format

**Location**: `~/.seismicx/config.json`

```json
{
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "sk-...",
    "temperature": 0.3,
    "max_tokens": 2000
  },
  "first_run": false
}
```

**Valid Providers**:
- `ollama`
- `deepseek`
- `openai`
- `siliconflow`
- `moonshot`
- `dashscope`
- `zhipu`
- `anthropic`
- `custom`

---

## 14. Summary: What Works vs What Doesn't

### ✅ Working
- Ollama local model detection and downloading
- Configuration persistence
- API key storage (masked in responses)
- Test connection for Ollama
- OpenAI-compatible API client implementation
- Automatic fallback to rule-based system
- DeepSeek API basic support
- Conversational agent with LLM backing

### ⚠️ Partially Working
- Online API configuration (saves but untested before saving)
- DashScope/Alibaba (no model list detection)
- Model name validation (only at runtime)

### ❌ Not Working / Missing
- Online API model list fetching via Web UI
- Web API endpoint for `/api/llm/online/models`
- Model dropdown on web UI
- API key validation before saving
- Detailed error messages for API connection failures
- Dynamic model list population on UI

---

## Conclusion

The SeismicX LLM architecture is **well-designed** with a clean abstraction layer that supports both local (Ollama) and online (OpenAI-compatible) APIs. The recent modifications (Apr 28, 2026) added solid support for 8+ online providers.

**Main Gap**: The **Web UI does not expose model list detection**, forcing users to either:
1. Use Python CLI: `python test_online_api_models.py deepseek sk-xxx`
2. Manually enter model names
3. Check provider documentation

**Next Step**: Implement the P0 items above to complete the online API integration.

