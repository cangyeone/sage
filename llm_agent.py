#!/usr/bin/env python3
"""
SeismicX LLM-driven Conversational Agent (Ollama tool-use)

把现有的地震学 skill 注册为 LLM 可以调用的 tools，然后让本地 Ollama 模型
（qwen2.5、llama3.1/3.2、mistral-nemo 等支持 tool calling 的模型）在多轮
对话中自主决定：
  1. 是不是要调 skill，调哪个；
  2. 参数怎么从用户的话 + 历史上下文里抠出来；
  3. 什么时候还需要追问用户。

对外暴露 ``OllamaToolAgent``，由 ``conversational_agent.ConversationalAgent``
包一层，保持 ``process_message(str) -> dict`` 的既有接口不变。
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error


# ------------------------------------------------------------------
# Tool schemas (OpenAI function-calling 风格，Ollama 直接兼容)
# ------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "browse_seismic_data",
            "description": (
                "扫描一个目录，列出其中的地震数据文件（mseed/sac/seed/miniseed）。"
                "当用户想‘查看/浏览/列出/看看’某目录下有什么地震数据时调用。"
                "返回的文件列表会被记住，之后用户说‘第N个’或‘刚才那个’时可以用 file_index 指代。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "要扫描的目录的绝对路径。"
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_waveform",
            "description": (
                "使用 ObsPy + matplotlib 绘制单个地震波形文件的时程图并保存 PNG。"
                "可以用绝对路径 file_path 指定文件，也可以用 file_index "
                "引用 browse_seismic_data 刚刚列出的文件（从 1 开始）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要绘制的地震数据文件绝对路径，与 file_index 二选一。"
                    },
                    "file_index": {
                        "type": "integer",
                        "description": "引用上一次 browse_seismic_data 返回列表中的文件序号（1-based）。"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pick_phases",
            "description": (
                "对一个目录下的波形做震相检测（Pg/Sg/Pn/Sn），"
                "调用 pnsn/picker.py。生成一个 txt 拾取文件，路径会被记住，"
                "后续的 associate_phases / analyze_polarity 可以直接复用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_dir": {
                        "type": "string",
                        "description": "包含波形文件的目录绝对路径。"
                    },
                    "model": {
                        "type": "string",
                        "description": "拾取模型路径，默认为 pnsn/pickers/pnsn.v3.jit。"
                    },
                    "output": {
                        "type": "string",
                        "description": "输出文件基名（不含扩展名），默认自动生成时间戳。"
                    }
                },
                "required": ["input_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "associate_phases",
            "description": (
                "把震相拾取结果关联成地震事件。如果用户没提供 input_file，"
                "可以不传，系统会自动使用上一次 pick_phases 的输出。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "拾取文件路径；不传则使用最近一次拾取结果。"
                    },
                    "station_file": {
                        "type": "string",
                        "description": "台站信息文件绝对路径（必填）。"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["fastlink", "reallink"],
                        "description": "关联方法，默认 fastlink。"
                    },
                    "output": {
                        "type": "string",
                        "description": "输出文件路径，默认自动生成。"
                    }
                },
                "required": ["station_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_polarity",
            "description": (
                "分析 P 波初动极性。input_file 不传则默认用最近一次拾取结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "拾取文件路径；不传则使用最近一次拾取结果。"
                    },
                    "waveform_dir": {
                        "type": "string",
                        "description": "波形文件目录绝对路径（必填）。"
                    }
                },
                "required": ["waveform_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_context",
            "description": (
                "查看当前对话里已记住的关键状态：上次浏览的目录、浏览到的文件列表、"
                "最近一次拾取输出、最近一次绘制的波形图等。"
                "当用户含糊地说‘刚才那个’‘上次的’但你又不确定指什么时调用，先看看再回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


SYSTEM_PROMPT = """你是 SeismicX 的地震学分析助手，基于本地 Ollama 运行，目标是通过多轮对话驱动以下地震数据处理任务：
- 浏览目录里的地震数据文件（mseed/sac/seed）
- 绘制波形图
- 震相检测（Pg/Sg/Pn/Sn）
- 震相关联 / 地震事件定位
- P 波初动极性分析

工作规则：
1. 仔细阅读整个对话历史，理解代词（"那个"、"刚才的"、"它"、"第 1 个"）指代的是什么。
   - 如果不确定指代对象，调用 recall_context 查看当前会话里的状态，而不是盲猜。
2. 只要用户意图能对应到一个工具，就直接调用工具；不要在文本里假装已经执行了。
3. 工具需要的信息不全时，用一句简短的中文向用户追问，不要凭空捏造路径。
4. 调用工具后，把工具返回里的 message 字段用自然的中文总结给用户；如果工具返回 success=false，
   如实告诉用户错误原因，并给出下一步建议。
5. 对于"接着做"、"继续"、"顺便"这类顺承式指令，主动基于前一步结果选择合适的工具和参数。
6. 回答保持简洁、专业，用中文。除非用户明确要求，不要大段罗列 Markdown。"""


# ------------------------------------------------------------------
# Tool backend: 把调用转给已有的 SkillExecutor
# ------------------------------------------------------------------

class ToolBackend:
    """实际执行工具调用，结果里尽量带上结构化数据给 LLM 当下一步素材。"""

    def __init__(self, skill_executor, context):
        self.skill_executor = skill_executor
        self.context = context

    # -- helpers -----------------------------------------------------

    def _auto_run_command(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """对于需要跑外部命令的 skill，同步执行并把 stdout/stderr 附回来。"""
        if result.get("action") != "execute_command" or "command" not in result:
            return result
        import subprocess
        cmd = result["command"]
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=1800
            )
            result["command_executed"] = True
            result["returncode"] = proc.returncode
            # stdout/stderr 很长时截断，避免把 LLM context 撑爆
            result["stdout_tail"] = (proc.stdout or "")[-2000:]
            result["stderr_tail"] = (proc.stderr or "")[-2000:]
            if proc.returncode != 0:
                result["success"] = False
                result["message"] = (
                    f"命令执行失败（returncode={proc.returncode}）:\n"
                    + (proc.stderr or proc.stdout or "(无输出)")[-800:]
                )
        except Exception as e:
            result["success"] = False
            result["message"] = f"命令执行异常: {e}"
        return result

    # -- tool implementations ----------------------------------------

    def browse_seismic_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        directory = args.get("directory", "")
        result = self.skill_executor._execute_data_browsing(
            {"file_paths": [directory]}, self.context
        )
        # 精简 message，把结构化数据补齐
        files = result.get("results", {}).get("files", [])
        result["file_index_map"] = [
            {"index": i + 1, "path": p, "name": Path(p).name}
            for i, p in enumerate(files[:50])
        ]
        return result

    def plot_waveform(self, args: Dict[str, Any]) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        if args.get("file_path"):
            entities["file_paths"] = [args["file_path"]]
        if args.get("file_index") is not None:
            try:
                entities["numbers"] = [float(args["file_index"])]
            except (TypeError, ValueError):
                pass
        result = self.skill_executor._execute_waveform_plotting(entities, self.context)
        if result.get("success") and result.get("results", {}).get("image_path"):
            self.context.last_results["last_plotted_file"] = (
                result["results"].get("source_file")
            )
        return result

    def pick_phases(self, args: Dict[str, Any]) -> Dict[str, Any]:
        entities: Dict[str, Any] = {
            "file_paths": [args["input_dir"]],
        }
        if args.get("model"):
            entities["model"] = args["model"]
        if args.get("output"):
            entities["output"] = args["output"]
        result = self.skill_executor._execute_phase_picking(entities, self.context)
        result = self._auto_run_command(result)
        if result.get("success") and result.get("results", {}).get("output_file"):
            self.context.last_results["picks_file"] = result["results"]["output_file"]
        return result

    def associate_phases(self, args: Dict[str, Any]) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        if args.get("input_file"):
            entities["input_file"] = args["input_file"]
        if args.get("station_file"):
            entities["station_file"] = args["station_file"]
        if args.get("method"):
            entities["method"] = args["method"]
        if args.get("output"):
            entities["output"] = args["output"]
        result = self.skill_executor._execute_phase_association(entities, self.context)
        result = self._auto_run_command(result)
        if result.get("success") and result.get("results", {}).get("output_file"):
            self.context.last_results["events_file"] = result["results"]["output_file"]
        return result

    def analyze_polarity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        if args.get("input_file"):
            entities["input_file"] = args["input_file"]
        if args.get("waveform_dir"):
            entities["waveform_dir"] = args["waveform_dir"]
        result = self.skill_executor._execute_polarity_analysis(entities, self.context)
        result = self._auto_run_command(result)
        if result.get("success") and result.get("results", {}).get("output_file"):
            self.context.last_results["polarity_file"] = result["results"]["output_file"]
        return result

    def recall_context(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        lr = self.context.last_results
        files = lr.get("browse_files", [])
        snapshot = {
            "current_task": self.context.current_task,
            "task_state": self.context.task_state,
            "browse_directory": lr.get("browse_directory"),
            "browse_files_count": len(files),
            "browse_files_preview": [
                {"index": i + 1, "name": Path(p).name}
                for i, p in enumerate(files[:10])
            ],
            "picks_file": lr.get("picks_file"),
            "events_file": lr.get("events_file"),
            "polarity_file": lr.get("polarity_file"),
            "last_plotted_file": lr.get("last_plotted_file"),
        }
        return {"success": True, "message": "当前会话状态快照", "results": snapshot}

    # -- dispatch ----------------------------------------------------

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, name, None)
        if not handler:
            return {
                "success": False,
                "message": f"未知工具: {name}",
            }
        try:
            return handler(args or {})
        except Exception as e:
            return {
                "success": False,
                "message": f"工具 {name} 执行异常: {e}",
                "traceback": traceback.format_exc()[-1200:],
            }


# ------------------------------------------------------------------
# Ollama 客户端（纯 urllib，不引入新依赖）
# ------------------------------------------------------------------

class OllamaClient:
    def __init__(self, api_base: str, model: str, temperature: float = 0.3,
                 timeout: float = 180.0):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.api_base}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False


class OpenAICompatibleClient:
    """支持 DeepSeek、Qwen、SiliconFlow 等 OpenAI-compatible API 的客户端。"""
    
    def __init__(self, api_base: str, model: str, api_key: str, 
                 temperature: float = 0.3, timeout: float = 180.0):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """调用 OpenAI-compatible API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2000,
        }
        if tools:
            # OpenAI API 使用 tools 字段
            payload["tools"] = [
                {
                    "type": "function",
                    "function": t.get("function", t)
                } for t in tools
            ]

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        
        body = json.loads(raw)
        
        # 统一返回格式，兼容 Ollama 的响应格式
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        # 处理 tool_calls（OpenAI 格式）
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                if tc.get("type") == "function":
                    tool_calls.append({
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"].get("arguments", "{}"),
                        }
                    })
        
        return {
            "message": {
                "content": message.get("content", ""),
                "tool_calls": tool_calls if tool_calls else None,
            }
        }

    def ping(self) -> bool:
        """测试 API 连接。"""
        try:
            # 尝试调用 /v1/models 端点
            req = urllib.request.Request(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False
    
    @staticmethod
    def list_models(api_base: str, api_key: str) -> Optional[List[str]]:
        """获取在线 API 的可用模型列表。"""
        try:
            req = urllib.request.Request(
                f"{api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            
            models = []
            # 不同 API 的模型列表格式可能不同
            data = body.get("data", [])
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("id")
                        if model_id:
                            models.append(model_id)
            
            return models if models else None
        except Exception:
            return None


# ------------------------------------------------------------------
# 主 Agent
# ------------------------------------------------------------------

class OllamaToolAgent:
    """多轮 tool-use 对话 Agent。支持 Ollama 和 OpenAI-compatible API。"""

    MAX_TOOL_ITERATIONS = 6  # 单次用户消息内，LLM 最多连续调用多少轮工具

    def __init__(self, skill_executor, context,
                 client = None,  # OllamaClient 或 OpenAICompatibleClient 实例
                 api_base: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 temperature: float = 0.3,
                 api_key: str = "",
                 max_history_messages: int = 40):
        # 如果没有传入 client，则根据参数创建 OllamaClient（向后兼容）
        if client is None:
            self.client = OllamaClient(api_base=api_base, model=model,
                                       temperature=temperature)
        else:
            self.client = client
        
        self.backend = ToolBackend(skill_executor, context)
        self.context = context
        self.max_history_messages = max_history_messages
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # -- conversation management -------------------------------------

    def _trim_history(self):
        """只保留 system + 最近 N 条，避免 context 无限膨胀。"""
        if len(self.messages) <= self.max_history_messages + 1:
            return
        system = self.messages[0]
        tail = self.messages[-self.max_history_messages:]
        # 工具消息一定要紧跟在触发它的 assistant tool_calls 后面，
        # 为了稳健，trim 时从最前面往后找第一个不是 tool 的消息作为新起点。
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        self.messages = [system] + tail

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # -- main entry --------------------------------------------------

    def process_message(self, user_message: str) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": user_message})
        last_tool_result: Dict[str, Any] = {}
        last_action = "none"

        for _ in range(self.MAX_TOOL_ITERATIONS):
            try:
                resp = self.client.chat(self.messages, tools=TOOL_SCHEMAS)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                raise OllamaUnavailable(str(e))
            except Exception as e:
                raise OllamaUnavailable(f"Ollama 响应解析失败: {e}")

            assistant_msg = resp.get("message", {}) or {}
            tool_calls = assistant_msg.get("tool_calls") or []

            # 无论是否 tool_calls，都把 assistant 回合塞回历史（保证后续
            # tool message 上下文对齐）
            stored_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.get("content", "") or "",
            }
            if tool_calls:
                stored_msg["tool_calls"] = tool_calls
            self.messages.append(stored_msg)

            if not tool_calls:
                final_text = (assistant_msg.get("content") or "").strip()
                if not final_text:
                    final_text = "（模型未返回内容，请换种问法再试。）"
                self._trim_history()
                return {
                    "response": final_text,
                    "action": last_action,
                    "data": last_tool_result,
                }

            # 执行所有工具调用
            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {"__raw__": raw_args}
                else:
                    args = raw_args or {}

                tool_result = self.backend.call(name, args)
                last_tool_result = tool_result
                last_action = tool_result.get("action", last_action)

                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False,
                                          default=str),
                })

            # 继续下一轮，让模型根据工具结果回答或再调工具

        # 超过最大工具轮次
        fallback = "工具调用链太长了，我先停下来。可否把需求再拆成更具体的一步？"
        self.messages.append({"role": "assistant", "content": fallback})
        self._trim_history()
        return {
            "response": fallback,
            "action": last_action,
            "data": last_tool_result,
        }


class OllamaUnavailable(RuntimeError):
    """Ollama 服务不可用或模型不支持 tool calling 时抛出，触发降级。"""


# ------------------------------------------------------------------
# 工厂：从 config_manager 读取配置
# ------------------------------------------------------------------

def build_agent_from_config(skill_executor, context) -> Optional[OllamaToolAgent]:
    """根据 ~/.seismicx/config.json 构建 agent；不可用时返回 None。
    
    支持 Ollama 和 OpenAI-compatible 在线 API（DeepSeek、Qwen、SiliconFlow 等）。
    """
    try:
        from config_manager import get_config_manager
    except ImportError:
        return None

    cfg = get_config_manager().get_llm_config()
    provider = (cfg.get("provider") or "").lower()
    model = cfg.get("model") or ""
    api_base = cfg.get("api_base") or "http://localhost:11434"
    temperature = float(cfg.get("temperature", 0.3))
    api_key = cfg.get("api_key") or ""

    # ──────────────────────────────────────────────────────────────
    # 1. Ollama 分支
    # ──────────────────────────────────────────────────────────────
    if provider == "ollama":
        if not model:
            model = "qwen2.5:7b"
        
        client = OllamaClient(api_base=api_base, model=model,
                             temperature=temperature)
        agent = OllamaToolAgent(
            skill_executor=skill_executor,
            context=context,
            client=client,
        )

        # 环境变量可以关掉 LLM 路径（调试/降级用）
        if os.environ.get("SEISMICX_DISABLE_LLM") == "1":
            return None

        # 如果 Ollama 服务都连不上，直接返回 None 让上层走规则引擎
        if not client.ping():
            return None

        return agent

    # ──────────────────────────────────────────────────────────────
    # 2. 在线 API 分支（DeepSeek、Qwen、OpenAI 等 OpenAI-compatible）
    # ──────────────────────────────────────────────────────────────
    elif provider in ["deepseek", "openai", "siliconflow", "moonshot", 
                       "dashscope", "zhipu", "anthropic", "custom"]:
        if not model:
            # 根据 provider 设置默认模型
            defaults = {
                "deepseek": "deepseek-v4-flash",
                "openai": "gpt-4o-mini",
                "siliconflow": "Qwen/Qwen2.5-7B-Instruct",
                "moonshot": "moonshot-v1-8k",
                "dashscope": "qwen-turbo",
                "zhipu": "glm-4-flash",
                "anthropic": "claude-3-5-haiku-20241022",
            }
            model = defaults.get(provider, "")
        
        if not model or not api_key:
            # 缺少必要配置
            return None
        
        client = OpenAICompatibleClient(
            api_base=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
        
        agent = OllamaToolAgent(
            skill_executor=skill_executor,
            context=context,
            client=client,
        )

        # 环境变量可以关掉 LLM 路径
        if os.environ.get("SEISMICX_DISABLE_LLM") == "1":
            return None

        # 在线 API 不通过 ping() 决定是否启用 agent。
        # ping() 调用 /models 端点，部分平台（如 DashScope）不支持该端点，
        # 会导致有效配置被误判为不可用。实际连接问题会在 chat 时暴露并报错。
        return agent

    # 其它 provider 目前不支持
    return None
