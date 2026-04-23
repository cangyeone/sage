#!/usr/bin/env python3
"""
backend_manager.py — SAGE LLM 后端管理器

支持三种后端：
  1. Ollama    — 本地运行，管理最简单
  2. vLLM      — 高性能本地推理，支持更多模型格式
  3. 在线 API  — OpenAI / DeepSeek / SiliconFlow / Moonshot 等

主要功能：
  - 自动检测当前可用后端
  - Ollama 未安装时引导安装或自动切换到 vLLM
  - vLLM 未安装时自动 pip install
  - 引导用户下载模型到固定目录
  - 在线 API 配置与验证
  - 统一的 get_llm_config() 接口，下游代码无需感知后端类型
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── 常量 ─────────────────────────────────────────────────────────────────────

# 本地模型统一存放目录
MODEL_DIR = Path.home() / ".seismicx" / "models"

# vLLM 默认端口
VLLM_DEFAULT_PORT = 8001

# 各在线服务商预设
ONLINE_PROVIDERS: Dict[str, Dict] = {
    "openai": {
        "display": "OpenAI (GPT-4o / GPT-4o-mini)",
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "need_key": True,
    },
    "deepseek": {
        "display": "DeepSeek (deepseek-chat / deepseek-coder)",
        "api_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "need_key": True,
    },
    "siliconflow": {
        "display": "SiliconFlow (免费额度，Qwen/DeepSeek 等)",
        "api_base": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "need_key": True,
    },
    "moonshot": {
        "display": "Moonshot / Kimi (moonshot-v1-8k)",
        "api_base": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "need_key": True,
    },
    "dashscope": {
        "display": "阿里云百炼 / DashScope (qwen-turbo)",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-turbo",
        "need_key": True,
    },
    "zhipu": {
        "display": "智谱 AI (glm-4-flash 免费)",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "need_key": True,
    },
    "anthropic": {
        "display": "Anthropic (Claude 3.5)",
        "api_base": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-haiku-20241022",
        "need_key": True,
    },
    "custom": {
        "display": "自定义 OpenAI 兼容接口",
        "api_base": "",
        "default_model": "",
        "need_key": False,
    },
}

# 推荐本地模型（适合地震学研究的中英文模型）
RECOMMENDED_LOCAL_MODELS: List[Dict] = [
    {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "ms_id": "qwen/Qwen2.5-7B-Instruct",
        "display": "Qwen2.5-7B-Instruct",
        "size": "~14 GB (bf16) / ~4 GB (Q4)",
        "desc": "强烈推荐：中英文双语，代码能力强，适合大多数 GPU",
        "ollama_tag": "qwen2.5:7b",
    },
    {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "ms_id": "qwen/Qwen2.5-14B-Instruct",
        "display": "Qwen2.5-14B-Instruct",
        "size": "~28 GB (bf16) / ~8 GB (Q4)",
        "desc": "更强推理能力，需 16 GB+ 显存",
        "ollama_tag": "qwen2.5:14b",
    },
    {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "ms_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "display": "DeepSeek-R1-Distill-7B",
        "size": "~14 GB (bf16) / ~4 GB (Q4)",
        "desc": "推理能力突出，适合复杂计划任务",
        "ollama_tag": "deepseek-r1:7b",
    },
    {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "ms_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "display": "Llama-3.1-8B-Instruct",
        "size": "~16 GB (bf16) / ~4.7 GB (Q4)",
        "desc": "英文能力强，开源社区活跃",
        "ollama_tag": "llama3.1:8b",
    },
]


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 5) -> Optional[Dict]:
    """GET 请求，返回 JSON 或 None。"""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _http_post(url: str, payload: Dict, api_key: str = "", timeout: int = 15) -> Optional[Dict]:
    """POST 请求，返回 JSON 或 None。"""
    try:
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _run(cmd: List[str], timeout: int = 30, capture: bool = True) -> Tuple[int, str, str]:
    """运行子进程，返回 (returncode, stdout, stderr)。"""
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", "not found"
    except Exception as e:
        return -1, "", str(e)


# ── 状态数据类 ────────────────────────────────────────────────────────────────

@dataclass
class OllamaStatus:
    installed: bool = False        # CLI 存在
    running: bool = False          # HTTP 服务可达
    models: List[str] = field(default_factory=list)
    api_base: str = "http://localhost:11434"


@dataclass
class VllmStatus:
    installed: bool = False        # pip 包已安装
    running: bool = False          # HTTP 服务可达
    pid: Optional[int] = None
    port: int = VLLM_DEFAULT_PORT
    model_path: str = ""
    api_base: str = f"http://localhost:{VLLM_DEFAULT_PORT}"


@dataclass
class OnlineStatus:
    provider: str = ""
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    reachable: bool = False


# ── 核心 BackendManager ───────────────────────────────────────────────────────

class BackendManager:
    """
    统一管理 Ollama / vLLM / 在线 API 三种后端。

    使用方法
    --------
    mgr = BackendManager()
    mgr.status()                      # 打印全状态
    mgr.auto_select()                 # 自动选择可用后端
    cfg = mgr.get_llm_config()        # 获取当前后端的 LLM 配置
    """

    def __init__(self):
        self.config_dir = Path.home() / ".seismicx"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
        self._vllm_proc: Optional[subprocess.Popen] = None

    # ── 配置读写 ──────────────────────────────────────────────────────────────

    def _load_config(self) -> Dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_config()

    def _default_config(self) -> Dict:
        return {
            "active_backend": "ollama",   # ollama | vllm | online
            "ollama": {
                "api_base": "http://localhost:11434",
                "model": "qwen2.5:7b",
            },
            "vllm": {
                "port": VLLM_DEFAULT_PORT,
                "model_path": "",         # 空 = 尚未配置
                "model": "",
                "gpu_memory_fraction": 0.9,
                "extra_args": [],
            },
            "online": {
                "provider": "deepseek",
                "api_base": ONLINE_PROVIDERS["deepseek"]["api_base"],
                "api_key": "",
                "model": ONLINE_PROVIDERS["deepseek"]["default_model"],
            },
            "first_run": True,
        }

    def _save(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    @property
    def active_backend(self) -> str:
        return self._config.get("active_backend", "ollama")

    # ── 后端检测 ──────────────────────────────────────────────────────────────

    def check_ollama(self) -> OllamaStatus:
        """检测 Ollama 状态。"""
        st = OllamaStatus()
        # 1. CLI
        rc, out, _ = _run(["ollama", "--version"])
        st.installed = (rc == 0)

        # 2. HTTP
        api_base = self._config.get("ollama", {}).get("api_base", "http://localhost:11434")
        st.api_base = api_base
        data = _http_get(f"{api_base}/api/tags", timeout=3)
        if data:
            st.running = True
            st.models = [m["name"] for m in data.get("models", [])]

        return st

    def check_vllm(self) -> VllmStatus:
        """检测 vLLM 状态。"""
        st = VllmStatus()
        vcfg = self._config.get("vllm", {})
        port = vcfg.get("port", VLLM_DEFAULT_PORT)
        st.port = port
        st.api_base = f"http://localhost:{port}"
        st.model_path = vcfg.get("model_path", "")
        st.model = vcfg.get("model", "")

        # 1. pip 包
        rc, out, _ = _run([sys.executable, "-c", "import vllm; print(vllm.__version__)"])
        st.installed = (rc == 0)

        # 2. HTTP 服务（OpenAI-compatible）
        data = _http_get(f"{st.api_base}/v1/models", timeout=3)
        if data:
            st.running = True

        return st

    def check_online(self) -> OnlineStatus:
        """检测在线 API 配置与可达性。"""
        ocfg = self._config.get("online", {})
        st = OnlineStatus(
            provider=ocfg.get("provider", ""),
            api_base=ocfg.get("api_base", ""),
            api_key=ocfg.get("api_key", ""),
            model=ocfg.get("model", ""),
        )
        if not st.api_key or not st.api_base:
            return st

        # 发一个极短的测试请求
        url = st.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": st.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        result = _http_post(url, payload, st.api_key, timeout=10)
        st.reachable = (result is not None and "choices" in result)
        return st

    def detect_all(self) -> Dict[str, Any]:
        """一次性检测全部后端，返回汇总字典。"""
        return {
            "ollama": self.check_ollama(),
            "vllm": self.check_vllm(),
            "online": self.check_online(),
        }

    # ── 安装 vLLM ─────────────────────────────────────────────────────────────

    def install_vllm(self, progress_cb=print, cpu_only: bool = False) -> bool:
        """
        自动安装 vLLM。

        Parameters
        ----------
        progress_cb : callable
            进度回调，接受字符串。
        cpu_only : bool
            True 时安装 CPU-only 版本（性能较低，但无需 GPU）。
        """
        progress_cb("正在检查 pip 环境…")
        rc, out, err = _run([sys.executable, "-m", "pip", "--version"])
        if rc != 0:
            progress_cb("✗ pip 不可用，请先安装 pip")
            return False

        # 检查 CUDA
        has_cuda = False
        rc2, out2, _ = _run([sys.executable, "-c",
                              "import torch; print(torch.cuda.is_available())"])
        if rc2 == 0 and "True" in out2:
            has_cuda = True

        if cpu_only or not has_cuda:
            progress_cb("⚠ 未检测到 CUDA GPU，将安装 CPU 版本（速度较慢）")
            cmd = [sys.executable, "-m", "pip", "install",
                   "vllm-cpu", "--break-system-packages", "-q"]
        else:
            progress_cb("✓ 检测到 CUDA GPU，安装 GPU 版 vLLM…")
            cmd = [sys.executable, "-m", "pip", "install",
                   "vllm", "--break-system-packages", "-q"]

        progress_cb(f"执行: {' '.join(cmd)}")
        progress_cb("（首次安装可能需要数分钟，请耐心等待）")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    progress_cb(f"  {line}")
            proc.wait()
            if proc.returncode == 0:
                progress_cb("✓ vLLM 安装成功！")
                return True
            else:
                progress_cb("✗ vLLM 安装失败，请查看上方错误信息")
                return False
        except Exception as e:
            progress_cb(f"✗ 安装出错: {e}")
            return False

    # ── 模型管理 ──────────────────────────────────────────────────────────────

    def list_local_models(self) -> List[Path]:
        """列出 MODEL_DIR 下的所有模型目录。"""
        if not MODEL_DIR.exists():
            return []
        # 模型目录通常包含 config.json 或 *.gguf
        models = []
        for p in sorted(MODEL_DIR.iterdir()):
            if p.is_dir() and (
                (p / "config.json").exists()
                or list(p.glob("*.safetensors"))
                or list(p.glob("*.gguf"))
                or list(p.glob("*.bin"))
            ):
                models.append(p)
        return models

    def model_download_guide(self, model_info: Optional[Dict] = None) -> str:
        """
        返回模型下载教程文字。
        model_info: RECOMMENDED_LOCAL_MODELS 中的一项，None 时返回通用教程。
        """
        model_dir_str = str(MODEL_DIR)

        if model_info is None:
            # 通用教程
            lines = [
                "═" * 60,
                "  本地模型下载指南",
                "═" * 60,
                "",
                f"模型统一存放位置：{model_dir_str}",
                "",
                "推荐下载方式（选其一）：",
                "",
                "【方式一】HuggingFace（需要网络或代理）",
                "  pip install huggingface_hub",
                "  huggingface-cli download <模型ID> \\",
                f"      --local-dir {model_dir_str}/<模型名>",
                "",
                "【方式二】ModelScope（国内速度更快）",
                "  pip install modelscope",
                "  python -c \"",
                "  from modelscope import snapshot_download",
                f"  snapshot_download('<ms_model_id>', cache_dir='{model_dir_str}')",
                "  \"",
                "",
                "【方式三】Git LFS（适合 safetensors 大文件）",
                "  git lfs install",
                f"  git clone https://huggingface.co/<模型ID> {model_dir_str}/<模型名>",
                "",
                "下载完成后，运行：",
                "  sage backend use vllm --model <模型路径>",
            ]
        else:
            hf_id = model_info["hf_id"]
            ms_id = model_info["ms_id"]
            name = model_info["display"]
            dest = str(MODEL_DIR / name)
            lines = [
                "═" * 60,
                f"  下载 {name}",
                "═" * 60,
                "",
                f"存放位置：{dest}",
                f"模型大小：{model_info['size']}",
                "",
                "【HuggingFace（需代理）】",
                "  pip install huggingface_hub -q",
                f"  huggingface-cli download {hf_id} \\",
                f"      --local-dir \"{dest}\"",
                "",
                "【ModelScope（国内推荐）】",
                "  pip install modelscope -q",
                "  python3 -c \"",
                "  from modelscope import snapshot_download",
                f"  snapshot_download('{ms_id}',",
                f"      cache_dir='{str(MODEL_DIR)}')",
                "  \"",
                "",
                "下载完成后配置 vLLM：",
                f"  sage backend use vllm --model \"{dest}\"",
            ]

        return "\n".join(lines)

    def pull_ollama_model(self, model_tag: str, progress_cb=print) -> bool:
        """拉取 Ollama 模型。"""
        progress_cb(f"拉取 Ollama 模型: {model_tag}（可能需要较长时间）…")
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", model_tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    progress_cb(f"  {line}")
            proc.wait()
            if proc.returncode == 0:
                progress_cb(f"✓ 模型 {model_tag} 拉取成功！")
                return True
            else:
                progress_cb(f"✗ 拉取失败（code={proc.returncode}）")
                return False
        except FileNotFoundError:
            progress_cb("✗ Ollama 未安装，请先安装 Ollama")
            return False
        except Exception as e:
            progress_cb(f"✗ 出错: {e}")
            return False

    # ── 启动/停止 vLLM ───────────────────────────────────────────────────────

    def start_vllm(
        self,
        model_path: str,
        port: Optional[int] = None,
        gpu_memory_fraction: float = 0.9,
        extra_args: Optional[List[str]] = None,
        progress_cb=print,
    ) -> bool:
        """
        启动 vLLM OpenAI 兼容服务。

        Parameters
        ----------
        model_path : str
            本地模型路径或 HuggingFace 模型 ID。
        port : int
            监听端口，默认 VLLM_DEFAULT_PORT。
        """
        if not self.check_vllm().installed:
            progress_cb("✗ vLLM 未安装，请先运行 `sage backend install vllm`")
            return False

        port = port or self._config.get("vllm", {}).get("port", VLLM_DEFAULT_PORT)
        extra = extra_args or self._config.get("vllm", {}).get("extra_args", [])

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_fraction),
            "--trust-remote-code",
        ] + extra

        progress_cb(f"启动 vLLM 服务: 模型={model_path}, 端口={port}")
        progress_cb(f"命令: {' '.join(cmd)}")

        try:
            self._vllm_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            progress_cb(f"✗ 启动失败: {e}")
            return False

        # 等待服务就绪（最多 120 秒）
        progress_cb("等待 vLLM 服务就绪（最多 120 秒）…")
        api_base = f"http://localhost:{port}"
        for i in range(24):
            time.sleep(5)
            data = _http_get(f"{api_base}/v1/models", timeout=3)
            if data:
                progress_cb(f"✓ vLLM 服务已启动：{api_base}")
                # 保存配置
                model_name = Path(model_path).name if Path(model_path).exists() else model_path
                self._config["vllm"]["port"] = port
                self._config["vllm"]["model_path"] = model_path
                self._config["vllm"]["model"] = model_name
                self._config["active_backend"] = "vllm"
                self._save()
                return True
            progress_cb(f"  等待中… ({(i+1)*5}s)")

        progress_cb("✗ vLLM 服务启动超时，请检查日志")
        return False

    def stop_vllm(self) -> bool:
        """停止 vLLM 服务。"""
        if self._vllm_proc:
            self._vllm_proc.terminate()
            self._vllm_proc = None
            return True
        # 尝试用 pkill
        rc, _, _ = _run(["pkill", "-f", "vllm.entrypoints.openai"])
        return rc == 0

    # ── 后端切换 ──────────────────────────────────────────────────────────────

    def use_ollama(self, model: Optional[str] = None, api_base: Optional[str] = None):
        """切换到 Ollama 后端。"""
        cfg = self._config.setdefault("ollama", {})
        if api_base:
            cfg["api_base"] = api_base
        if model:
            cfg["model"] = model
        self._config["active_backend"] = "ollama"
        self._save()

    def use_vllm(self, model_path: str, port: Optional[int] = None):
        """切换到 vLLM 后端（不启动服务，只更新配置）。"""
        cfg = self._config.setdefault("vllm", {})
        cfg["model_path"] = model_path
        cfg["model"] = Path(model_path).name if Path(model_path).exists() else model_path
        if port:
            cfg["port"] = port
        port = cfg.get("port", VLLM_DEFAULT_PORT)
        self._config["active_backend"] = "vllm"
        self._save()

    def use_online(self, provider: str, api_key: str,
                   model: Optional[str] = None, api_base: Optional[str] = None):
        """切换到在线 API 后端。"""
        preset = ONLINE_PROVIDERS.get(provider, {})
        cfg = self._config.setdefault("online", {})
        cfg["provider"] = provider
        cfg["api_base"] = api_base or preset.get("api_base", "")
        cfg["api_key"] = api_key
        cfg["model"] = model or preset.get("default_model", "")
        self._config["active_backend"] = "online"
        self._save()

    # ── 获取 LLM 配置（下游统一入口）────────────────────────────────────────

    def get_llm_config(self) -> Dict:
        """
        返回当前激活后端的 LLM 配置字典，格式与 planner/_call_llm 兼容：
        {provider, model, api_base, api_key}
        """
        backend = self.active_backend

        if backend == "ollama":
            cfg = self._config.get("ollama", {})
            return {
                "provider": "ollama",
                "model": cfg.get("model", "qwen2.5:7b"),
                "api_base": cfg.get("api_base", "http://localhost:11434"),
                "api_key": "",
            }

        elif backend == "vllm":
            cfg = self._config.get("vllm", {})
            port = cfg.get("port", VLLM_DEFAULT_PORT)
            return {
                "provider": "openai",   # vLLM 兼容 OpenAI 协议
                "model": cfg.get("model", cfg.get("model_path", "")),
                "api_base": f"http://localhost:{port}/v1",
                "api_key": "EMPTY",     # vLLM 默认不验证 key
            }

        elif backend == "online":
            cfg = self._config.get("online", {})
            return {
                "provider": cfg.get("provider", "openai"),
                "model": cfg.get("model", ""),
                "api_base": cfg.get("api_base", ""),
                "api_key": cfg.get("api_key", ""),
            }

        # 兜底
        return {"provider": "ollama", "model": "qwen2.5:7b",
                "api_base": "http://localhost:11434", "api_key": ""}

    # ── 自动选择 ──────────────────────────────────────────────────────────────

    def auto_select(self, progress_cb=print) -> str:
        """
        自动探测并选择第一个可用后端。
        优先级: Ollama(running) > vLLM(running) > online(configured) > Ollama(installed)
        返回选中的后端名称。
        """
        all_st = self.detect_all()

        if all_st["ollama"].running:
            self.use_ollama()
            progress_cb("✓ 自动选择后端: Ollama（服务运行中）")
            return "ollama"

        if all_st["vllm"].running:
            self.use_vllm(all_st["vllm"].model_path)
            progress_cb("✓ 自动选择后端: vLLM（服务运行中）")
            return "vllm"

        if all_st["online"].reachable:
            progress_cb("✓ 自动选择后端: 在线 API")
            self._config["active_backend"] = "online"
            self._save()
            return "online"

        progress_cb("⚠ 未检测到任何可用后端，保持当前配置")
        return self.active_backend

    # ── 状态打印 ──────────────────────────────────────────────────────────────

    def print_status(self):
        """打印所有后端的当前状态。"""
        all_st = self.detect_all()
        active = self.active_backend

        print()
        print("═" * 58)
        print("  SAGE 后端状态")
        print("═" * 58)

        # Ollama
        ost = all_st["ollama"]
        mark = "▶" if active == "ollama" else " "
        run = "✓ 运行中" if ost.running else ("已安装" if ost.installed else "✗ 未安装")
        print(f"\n{mark} [Ollama]  {run}")
        if ost.running and ost.models:
            print(f"    可用模型: {', '.join(ost.models[:5])}")
        elif not ost.installed:
            print("    安装: https://ollama.ai")

        # vLLM
        vst = all_st["vllm"]
        mark = "▶" if active == "vllm" else " "
        if vst.running:
            run = f"✓ 运行中 (port {vst.port})"
        elif vst.installed:
            run = "已安装（未启动）"
        else:
            run = "✗ 未安装"
        print(f"\n{mark} [vLLM]   {run}")
        local_models = self.list_local_models()
        if local_models:
            print(f"    本地模型 ({MODEL_DIR}):")
            for m in local_models[:4]:
                print(f"      · {m.name}")
        else:
            print(f"    模型目录: {MODEL_DIR}  (空)")

        # Online
        ost2 = all_st["online"]
        mark = "▶" if active == "online" else " "
        if ost2.reachable:
            run = f"✓ 可达  [{ost2.provider}] {ost2.model}"
        elif ost2.api_key:
            run = f"已配置（未验证）  [{ost2.provider}] {ost2.model}"
        else:
            run = "未配置（无 API Key）"
        print(f"\n{mark} [在线API] {run}")

        print()
        cfg = self.get_llm_config()
        print(f"当前生效配置: provider={cfg['provider']}, model={cfg['model']}")
        print(f"              api_base={cfg['api_base']}")
        print("═" * 58)
        print()

    # ── 交互式向导 ───────────────────────────────────────────────────────────

    def interactive_setup(self):
        """首次运行或手动触发的交互式配置向导。"""
        print()
        print("═" * 60)
        print("  SAGE 后端配置向导")
        print("═" * 60)

        all_st = self.detect_all()

        print("\n检测到的后端:")
        print(f"  Ollama : {'✓ 运行中' if all_st['ollama'].running else ('已安装' if all_st['ollama'].installed else '✗ 未安装')}")
        print(f"  vLLM   : {'✓ 运行中' if all_st['vllm'].running else ('已安装' if all_st['vllm'].installed else '✗ 未安装')}")
        print(f"  在线API: {'✓ 已配置' if all_st['online'].api_key else '未配置'}")

        print()
        print("请选择后端类型：")
        print("  1. Ollama   — 推荐，安装简单，自动管理模型")
        print("  2. vLLM     — 高性能，支持更多模型格式，需要 GPU")
        print("  3. 在线 API — DeepSeek / OpenAI / SiliconFlow 等")
        print()

        choice = input("选择 (1/2/3) [1]: ").strip() or "1"

        if choice == "1":
            self._wizard_ollama(all_st["ollama"])
        elif choice == "2":
            self._wizard_vllm(all_st["vllm"])
        elif choice == "3":
            self._wizard_online()
        else:
            print("无效选项，保持当前配置")
            return

        print()
        self.print_status()
        self._config["first_run"] = False
        self._save()

    def _wizard_ollama(self, ost: OllamaStatus):
        """Ollama 配置向导。"""
        if not ost.installed:
            print()
            print("Ollama 未安装，请先安装：")
            print("  macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh")
            print("  Windows    : 下载 https://ollama.ai")
            print("  安装后运行 : ollama serve")
            print()
            ok = input("安装完成后继续? (y/n) [n]: ").strip().lower()
            if ok != "y":
                return

        if not ost.running:
            print()
            print("Ollama 未运行，请在另一个终端执行：  ollama serve")
            input("启动后按 Enter 继续…")
            ost = self.check_ollama()
            if not ost.running:
                print("✗ 仍无法连接 Ollama，跳过")
                return

        # 选择模型
        print()
        print("已安装的模型:")
        if ost.models:
            for i, m in enumerate(ost.models, 1):
                print(f"  {i}. {m}  (已安装)")
        else:
            print("  （暂无）")

        print()
        print("推荐模型:")
        for i, m in enumerate(RECOMMENDED_LOCAL_MODELS, len(ost.models) + 1):
            print(f"  {i}. {m['ollama_tag']:30s}  {m['desc'][:40]}")

        print()
        raw = input("输入序号或模型名 [qwen2.5:7b]: ").strip() or "qwen2.5:7b"
        try:
            idx = int(raw) - 1
            all_tags = [m for m in ost.models] + [m["ollama_tag"] for m in RECOMMENDED_LOCAL_MODELS]
            model_tag = all_tags[idx] if 0 <= idx < len(all_tags) else raw
        except ValueError:
            model_tag = raw

        if model_tag not in ost.models:
            pull = input(f"模型 '{model_tag}' 未安装，立即拉取？(y/n) [y]: ").strip().lower()
            if pull != "n":
                self.pull_ollama_model(model_tag)

        self.use_ollama(model=model_tag)
        print(f"✓ 已切换到 Ollama，模型: {model_tag}")

    def _wizard_vllm(self, vst: VllmStatus):
        """vLLM 配置向导。"""
        if not vst.installed:
            print()
            inst = input("vLLM 未安装，现在自动安装？(y/n) [y]: ").strip().lower()
            if inst == "n":
                return
            cpu = input("是否安装 CPU-only 版本（无 GPU）？(y/n) [n]: ").strip().lower()
            ok = self.install_vllm(cpu_only=(cpu == "y"))
            if not ok:
                return

        # 选择模型
        local_models = self.list_local_models()
        print()
        print(f"本地模型目录: {MODEL_DIR}")
        if local_models:
            print("已有本地模型:")
            for i, p in enumerate(local_models, 1):
                print(f"  {i}. {p}")
        else:
            print("  （暂无模型）")
            print()
            print("推荐下载的模型:")
            for i, m in enumerate(RECOMMENDED_LOCAL_MODELS, 1):
                print(f"  {i}. {m['display']:30s}  {m['size']}  {m['desc'][:35]}")
            print()
            idx_raw = input("输入序号查看下载教程（或直接输入本地路径）: ").strip()
            try:
                idx = int(idx_raw) - 1
                if 0 <= idx < len(RECOMMENDED_LOCAL_MODELS):
                    print()
                    print(self.model_download_guide(RECOMMENDED_LOCAL_MODELS[idx]))
                    input("\n下载完成后按 Enter 继续…")
            except ValueError:
                pass

        # 重新扫描
        local_models = self.list_local_models()
        if not local_models:
            path_raw = input("直接输入模型路径（或 HuggingFace 模型 ID）: ").strip()
        else:
            print("当前本地模型:")
            for i, p in enumerate(local_models, 1):
                print(f"  {i}. {p}")
            path_raw = input("输入序号或模型路径: ").strip()
            try:
                idx = int(path_raw) - 1
                if 0 <= idx < len(local_models):
                    path_raw = str(local_models[idx])
            except ValueError:
                pass

        if not path_raw:
            print("✗ 未指定模型路径，跳过")
            return

        port_raw = input(f"vLLM 监听端口 [{VLLM_DEFAULT_PORT}]: ").strip()
        port = int(port_raw) if port_raw.isdigit() else VLLM_DEFAULT_PORT

        start = input(f"立即启动 vLLM 服务？(y/n) [y]: ").strip().lower()
        if start != "n":
            self.start_vllm(path_raw, port=port)
        else:
            self.use_vllm(path_raw, port=port)
            print(f"✓ 已配置 vLLM（未启动），模型: {path_raw}")
            print(f"  手动启动: sage backend start-vllm")

    def _wizard_online(self):
        """在线 API 配置向导。"""
        print()
        print("支持的在线服务商:")
        providers = list(ONLINE_PROVIDERS.keys())
        for i, key in enumerate(providers, 1):
            info = ONLINE_PROVIDERS[key]
            print(f"  {i}. {info['display']}")

        print()
        raw = input("选择序号 [2=DeepSeek]: ").strip() or "2"
        try:
            idx = int(raw) - 1
            provider = providers[idx] if 0 <= idx < len(providers) else "deepseek"
        except ValueError:
            provider = raw if raw in ONLINE_PROVIDERS else "deepseek"

        info = ONLINE_PROVIDERS[provider]
        print(f"\n已选: {info['display']}")
        if provider == "custom":
            api_base = input("输入 API Base URL: ").strip()
        else:
            api_base = info["api_base"]
            print(f"API Base: {api_base}")

        api_key = input("输入 API Key: ").strip()
        model_raw = input(f"模型名 [{info['default_model']}]: ").strip()
        model = model_raw or info["default_model"]

        self.use_online(provider, api_key, model=model, api_base=api_base)
        print()

        # 验证
        print("验证连接…")
        ocfg = self._config["online"]
        url = ocfg["api_base"].rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        result = _http_post(url, payload, api_key, timeout=15)
        if result and "choices" in result:
            print("✓ 连接成功！")
        else:
            print("⚠ 连接验证失败（API Key 或网络问题），配置已保存但请检查")


# ── 单例 ──────────────────────────────────────────────────────────────────────

_backend_manager: Optional[BackendManager] = None


def get_backend_manager() -> BackendManager:
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = BackendManager()
    return _backend_manager


if __name__ == "__main__":
    mgr = BackendManager()
    mgr.print_status()
