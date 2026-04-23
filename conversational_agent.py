#!/usr/bin/env python3
"""
SeismicX Conversational Agent

An intelligent dialogue system that allows users to control seismic analysis
skills through natural language conversation. Supports multi-turn interactions,
context awareness, and automatic skill routing.

Usage:
    python conversational_agent.py  # Interactive mode
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Optional imports for waveform visualization
try:
    import obspy
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np  # ensure np is always available
except ImportError:
    pass

# SagePicker — lazy import so missing deps don't break agent startup
_SagePicker = None
def _get_sage_picker_class():
    global _SagePicker
    if _SagePicker is None:
        try:
            import sys, os as _os
            _proj = str(Path(__file__).parent)
            if _proj not in sys.path:
                sys.path.insert(0, _proj)
            from pnsn.sage_picker import SagePicker
            _SagePicker = SagePicker
        except Exception:
            pass
    return _SagePicker


class ConversationContext:
    """Maintains context across multiple conversation turns"""

    def __init__(self):
        self.current_task: Optional[str] = None  # Current ongoing task
        self.task_state: Dict = {}  # Task-specific state
        self.last_results: Dict = {}  # Results from last operation
        self.user_preferences: Dict = {}  # User's preferred settings
        self.conversation_history: List[Dict] = []  # Full conversation log

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

    def get_recent_context(self, n_turns: int = 5) -> str:
        """Get recent conversation context for LLM"""
        messages = self.conversation_history[-n_turns * 2:]  # Last n turns
        context = ""
        for msg in messages:
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role_label}: {msg['content']}\n"
        return context


class IntentClassifier:
    """Classifies user intent from natural language input"""

    def __init__(self):
        # Intent patterns with keywords
        self.intent_patterns = {
            'batch_picking': {
                'keywords': ['遍历', '批量', '全部', '所有台站', '所有震相', '全部震相',
                             '目录下', 'batch', 'traverse', 'all stations'],
                'patterns': [
                    r'遍历.*拾取',
                    r'遍历.*目录',
                    r'批量.*拾取',
                    r'批量.*震相',
                    r'所有.*台站.*拾取',
                    r'拾取.*所有.*台站',
                    r'全部.*台站',
                    r'拾取.*整个.*目录',
                    r'traverse.*pick',
                    # 给定路径 + 目录下 + 所有/全部
                    r'拾取.*目录.*所有',
                    r'拾取.*目录.*震相',
                    r'目录下.*所有.*震相',
                    r'目录.*下.*所有',
                    r'所有震相',
                    r'全部震相',
                ]
            },
            'confirm_picking': {
                'keywords': ['跳过', 'skip', '复制', 'duplicate', '补齐', '继续'],
                'patterns': [
                    r'^跳过$', r'^skip$',
                    r'^复制$', r'^duplicate$',
                    r'直接跳过', r'复制.*补齐', r'补齐.*分量',
                    r'继续.*拾取',
                ]
            },
            'phase_picking': {
                'keywords': ['拾取', 'pick', '相位', '震相', 'phase'],
                'patterns': [
                    r'拾取.*震相',
                    r'detect.*phase',
                    r'pick.*phase',
                    r'检测.*震相',
                    r'检测.*Pg',
                    r'检测.*Sg',
                    r'检测.*相位',
                ]
            },
            'phase_association': {
                'keywords': ['关联', 'association', 'associate', '定位', 'locate', 'event'],
                'patterns': [
                    r'关联.*震相',
                    r'associate.*phase',
                    r'定位.*地震',
                    r'locate.*event',
                ]
            },
            'polarity_analysis': {
                'keywords': ['极性', 'polarity', '初动', 'first motion', 'focal'],
                'patterns': [
                    r'分析.*极性',
                    r'analyze.*polarity',
                    r'初动.*方向',
                ]
            },
            'status_check': {
                'keywords': ['状态', 'status', '进度', 'progress', '完成', 'done'],
                'patterns': [
                    r'处理.*完成',
                    r'is.*ready',
                    r'状态.*如何',
                ]
            },
            'help': {
                'keywords': ['帮助', 'help', '怎么用', 'how to', '可以做什么'],
                'patterns': [
                    r'怎么.*拾取',
                    r'how.*pick',
                    r'能.*什么',
                ]
            },
            'configure': {
                'keywords': ['配置', 'config', '设置', 'set', 'change', '修改'],
                'patterns': [
                    r'设置.*模型',
                    r'configure.*model',
                    r'修改.*参数',
                ]
            },
            'data_browsing': {
                'keywords': ['查看', '浏览', 'list', 'browse', 'find', '查找', '目录', '文件夹', '看下', '看看', '显示', '有哪些'],
                'patterns': [
                    r'查看.*目录',
                    r'browse.*directory',
                    r'list.*files',
                    r'有哪些.*文件',
                    r'目录下.*什么',
                    r'看下.*文件夹',
                    r'看看.*目录',
                    r'显示.*数据',
                ]
            },
            'waveform_plotting': {
                'keywords': ['绘制', 'plot', 'draw', '显示波形', 'waveform', '波形图', '画', '可视化'],
                'patterns': [
                    r'绘制.*波形',
                    r'plot.*waveform',
                    r'显示.*mseed',
                    r'画.*图',
                    r'可视化.*数据',
                    r'画一下.*文件',
                ]
            },
            'seismo_qa': {
                # 纯问答意图：用户在提问，不是要执行代码
                'keywords': [
                    '如何', '怎么', '什么是', '原理', '介绍', '解释',
                    '是什么', '有哪些', '为什么', '区别', '对比',
                    'how', 'what is', 'why', 'explain', 'difference',
                    '怎样', '能否介绍', '告诉我',
                ],
                'patterns': [
                    r'如何.{1,20}[？?]?$',
                    r'怎么.{1,20}[？?]?$',
                    r'怎样.{1,20}[？?]?$',
                    r'什么是',
                    r'.{1,20}是什么[？?]?$',
                    r'.{1,20}有哪些[？?]?$',
                    r'为什么',
                    r'能否.*介绍',
                    r'请.*介绍',
                    r'请.*解释',
                    r'what is\b',
                    r'how (to|do|does)\b',
                ]
            },
            'seismo_programming': {
                # 编程执行意图：用户明确要求生成/运行代码或处理数据
                # 关键词要求能区分"操作请求"和"知识问答"，避免使用单纯的技术术语
                'keywords': [
                    '代码', 'code', '编程', '实现', '写代码', '生成代码',
                    '去仪器响应', 'hvsr', '功率谱', 'psd', '粒子运动',
                    '去均值', '去趋势', 'detrend', '重采样', 'resample',
                ],
                'patterns': [
                    # 明确的帮我做XX动作
                    r'帮我.*画.*(图|谱|波形)',
                    r'帮我.*实现',
                    r'帮我.*(写|生成).*代码',
                    r'帮我.*对.*[做进行]',
                    r'帮我.*(处理|滤波|分析|计算).*(数据|波形|文件|mseed|sac)',
                    r'给(我|一个).*代码',
                    r'写.*代码',
                    r'生成.*代码',
                    # 对具体数据对象进行操作（需要有"对/把/将+对象"前缀）
                    r'对.{1,30}(做|进行|应用).*(滤波|filter|处理|去均值|去趋势|重采样)',
                    r'对.{1,30}(滤波|去仪器|去均值|去趋势|重采样)',
                    r'(对|把|将).{1,20}(做|进行|应用).{0,5}(bandpass|lowpass|highpass|带通|低通|高通)',
                    r'(bandpass|lowpass|highpass|带通|低通|高通).{0,10}(滤波|filter).{0,5}(这|该|此|文件|数据|波形)',
                    # 画图类（明确有数据文件或具体对象）
                    r'画出.*(波形|图|谱)',
                    r'绘制.*频谱',
                    r'绘制.*走时',
                    r'画走时.*曲线',
                    # 计算类（需要具体参数，不是概念问答）
                    r'计算.*ML|计算.*MW|计算.*震级.*km',
                    r'计算.*震源.*参数',
                    r'计算.*频谱',
                    # 工具函数/专业操作
                    r'去.*仪器响应',
                    r'去仪器',
                    r'plot.*spectrum',
                    r'compute.*magnitude',
                    r'走时.*曲线',
                    r'travel.*time.*curve',
                    r'taup|taupy',
                    r'频谱.*分析.*[这该]',
                    r'功率.*谱.*[这该]',
                    r'hvsr.*分析',
                    r'读取.*并.*[画绘]',
                    r'处理.*并.*[画绘输出]',
                ]
            },
            'tool_documentation': {
                'keywords': [
                    '文档', 'readme', 'manual', '说明', '教程',
                    'hypodd', 'velest', 'nonlinloc', 'hypoinverse', 'focmec',
                    '定位工具', '反演工具', '调用工具', '外部工具',
                    '生成输入', '输入文件', '运行工具',
                ],
                'patterns': [
                    r'根据.*文档',
                    r'读取.*文档',
                    r'给定.*说明',
                    r'用.*hypodd',
                    r'调用.*hypodd',
                    r'用.*velest',
                    r'用.*nonlinloc',
                    r'用.*hypoinverse',
                    r'用.*focmec',
                    r'运行.*定位',
                    r'生成.*输入文件',
                    r'工具.*文档',
                    r'文档.*工具',
                    r'有哪些.*工具',
                    r'支持.*工具',
                ]
            },
            'seismo_agent': {
                # 自主Agent意图：阅读文献 + 自主规划 + 编程实现
                'keywords': [
                    '阅读', '读取文献', '读论文', '文献', '论文', 'paper', 'pdf',
                    'arxiv', 'doi', '实现论文', '复现', '按照论文',
                    'agent', '自主', '规划', '多步',
                ],
                'patterns': [
                    r'(阅读|读取|分析).{0,10}(论文|文献|paper|pdf)',
                    r'(实现|复现|按照|根据).{0,10}(论文|文献|方法|算法)',
                    r'arxiv[:/]?\d{4}',
                    r'doi[:/]10\.',
                    r'\.pdf',
                    r'自主.{0,10}(规划|编程|实现)',
                    r'看完.{0,10}论文',
                ]
            },
            'seismo_statistics': {
                'keywords': [
                    'b值', 'b-value', 'bvalue', 'gutenberg', 'richter',
                    '完整性', 'completeness', 'Mc', '震级', '频率',
                    '统计', 'statistics', '分析', '频度', '地震目录',
                    'catalog', '时空分布', '空间分布', '时间分布',
                ],
                'patterns': [
                    r'计算.*b值',
                    r'b值.*计算',
                    r'b.*value',
                    r'gutenberg.*richter',
                    r'g-r.*关系',
                    r'gr.*关系',
                    r'频率.*震级.*关系',
                    r'计算.*Mc',
                    r'完整性.*震级',
                    r'震级.*完整性',
                    r'频率.*震级',
                    r'震级.*频率',
                    r'地震.*统计',
                    r'统计.*分析',
                    r'时空.*分布',
                    r'空间.*分布',
                    r'时间.*分布',
                    r'给定.*数据.*计算',
                    r'analyze.*catalog',
                    r'seismicity.*analysis',
                ]
            }
        }

    # ── LLM intent classification ──────────────────────────────────────────

    _INTENT_SYSTEM = """\
你是一个地震学分析系统的意图分类器。根据用户输入，判断其属于以下哪个意图，并以 JSON 返回。

意图列表（选择其中一个）：
- batch_picking      : 批量遍历目录对所有台站/文件进行震相拾取
- confirm_picking    : 确认/跳过/继续当前拾取操作（如"跳过"、"复制补齐"）
- phase_picking      : 对指定文件或台站进行震相拾取检测
- phase_association  : 震相关联、地震事件定位
- polarity_analysis  : 震相极性分析、震源机制
- status_check       : 查询当前任务状态或进度
- help               : 询问系统功能帮助
- configure          : 修改配置、切换模型参数
- data_browsing      : 查看/浏览/列出数据目录或文件
- waveform_plotting  : 绘制波形图（不需要生成新代码，直接画图）
- seismo_statistics  : 地震统计分析（b值、G-R关系、震级-频度分布等）
- seismo_qa          : 地震学知识问答（用户在提问，不需要执行代码，如"什么是P波"、"如何做带通滤波"）
- seismo_programming : 用户要求生成并执行 Python 代码处理数据（如"帮我对XX做带通滤波"、"写代码计算PSD"）
- seismo_agent       : 读取论文/文献，自主规划并编程实现算法（如"阅读这篇论文并实现"）
- tool_documentation : 解析外部工具文档或生成 HypoDD/VELEST/NonLinLoc 等工具的输入文件

关键区分：
- seismo_qa vs seismo_programming：用户在"问"如何做（qa）还是"让你去做"（programming）
  例：「如何做带通滤波？」→ seismo_qa；「帮我对数据做带通滤波」→ seismo_programming
- seismo_programming vs seismo_agent：单次代码任务（programming）还是需要读论文自主规划（agent）

输出格式（严格 JSON，无其他内容）：
{"intent": "<意图名>", "confidence": <0.0-1.0>}
"""

    def _classify_with_llm(self, user_input: str, llm_config: Dict) -> Optional[Dict]:
        """Call LLM for intent classification. Returns None on failure."""
        provider = llm_config.get("provider", "ollama")
        model = llm_config.get("model", "qwen2.5:7b")
        api_base = llm_config.get("api_base", "http://localhost:11434")
        api_key = llm_config.get("api_key", "")

        messages = [
            {"role": "system", "content": self._INTENT_SYSTEM},
            {"role": "user", "content": user_input},
        ]

        if provider == "ollama":
            url = api_base.rstrip("/") + "/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 64},
            }
        else:
            url = api_base.rstrip("/") + "/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 64,
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
        except Exception:
            return None

        if provider == "ollama":
            raw = body.get("message", {}).get("content", "")
        else:
            raw = body.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON response
        raw = re.sub(r"```json\s*|```", "", raw).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None
        try:
            result = json.loads(m.group())
            intent = result.get("intent", "").strip()
            confidence = float(result.get("confidence", 0.8))
            if intent in self.intent_patterns or intent == "unknown":
                return {"intent": intent, "confidence": confidence}
        except Exception:
            pass
        return None

    def _classify_with_rules(self, user_input: str) -> Dict:
        """Fallback rule-based classification (keyword + regex scoring)."""
        user_input_lower = user_input.lower()
        scores = {}

        for intent, config in self.intent_patterns.items():
            score = 0
            for keyword in config['keywords']:
                if keyword.lower() in user_input_lower:
                    score += 1
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower):
                    score += 2
            scores[intent] = score

        if not scores or max(scores.values()) == 0:
            return {'intent': 'unknown', 'confidence': 0.0}

        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 5.0, 1.0)
        return {'intent': best_intent, 'confidence': confidence}

    def classify(self, user_input: str, llm_config: Optional[Dict] = None) -> Dict:
        """
        Classify user intent — tries LLM first, falls back to rule-based.

        Returns: {intent: str, confidence: float, entities: dict, method: str}
        """
        # Load LLM config if not provided
        if llm_config is None:
            try:
                import sys as _sys
                _proj = str(Path(__file__).parent)
                if _proj not in _sys.path:
                    _sys.path.insert(0, _proj)
                from config_manager import LLMConfigManager
                llm_config = LLMConfigManager().get_llm_config()
            except Exception:
                llm_config = {}

        # 1. Try LLM classification
        llm_result = self._classify_with_llm(user_input, llm_config) if llm_config else None

        if llm_result:
            intent = llm_result["intent"]
            confidence = llm_result["confidence"]
            method = "llm"
        else:
            # 2. Fallback to rule-based
            rule_result = self._classify_with_rules(user_input)
            intent = rule_result["intent"]
            confidence = rule_result["confidence"]
            method = "rules"

        entities = self._extract_entities(user_input)
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'method': method,
        }

    def _extract_entities(self, text: str) -> Dict:
        """Extract entities like file paths, model names, etc."""
        entities = {}

        # Extract file paths
        # 说明：路径只允许 ASCII 可见字符中常见的合法路径字符
        # （字母、数字、/、.、_、-、~、:、空格被用 \s 屏蔽），
        # 这样可以避免把紧跟在路径后的中文（例如 "目录中的文件"）误识别为路径。
        path_pattern = r'(?:\.{1,2})?/[A-Za-z0-9_\-./~:]+'
        raw_paths = re.findall(path_pattern, text)
        # 去掉尾部可能误带的标点，例如逗号、句号、问号等
        cleaned_paths = [p.rstrip('.,;:!?，。；：！？、)】」』"\'') for p in raw_paths]
        # 过滤空字符串
        cleaned_paths = [p for p in cleaned_paths if p]
        if cleaned_paths:
            entities['file_paths'] = cleaned_paths

        # Extract model names
        model_patterns = [
            r'(pnsn\.v[\d.]+)',
            r'(phasenet)',
            r'(eqtransformer)',
            r'(llama[\d.]+)',
            r'(gpt-[\d.]+)',
        ]
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['model'] = match.group(1)
                break

        # Extract NET.STA station identifiers (e.g. X1.53085, XJ.KAS01)
        # Use lookahead/lookbehind instead of \b — \b fails adjacent to Chinese chars
        station_pattern = r'(?<![A-Za-z0-9])([A-Z0-9]{1,3})\.([A-Z0-9]{3,7})(?![A-Za-z0-9])'
        stations = re.findall(station_pattern, text, re.IGNORECASE)
        if stations:
            entities['stations'] = [f'{net}.{sta}'.upper() for net, sta in stations]

        # Extract numbers (thresholds, counts, etc.)
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            entities['numbers'] = [float(n) for n in numbers]

        return entities


class SkillExecutor:
    """Executes seismic analysis skills based on intent"""

    def __init__(self):
        self.available_skills = {
            'phase_picking': {
                'name': 'seismic-phase-picker',
                'description': 'Detect seismic phases from waveform data',
                'parameters': ['input_dir', 'output', 'model', 'device']
            },
            'phase_association': {
                'name': 'seismic-phase-associator',
                'description': 'Associate phase picks into earthquake events',
                'parameters': ['input_file', 'station_file', 'output', 'method']
            },
            'polarity_analysis': {
                'name': 'seismic-polarity-analyzer',
                'description': 'Analyze first-motion polarity of P-waves',
                'parameters': ['input_file', 'waveform_dir', 'output']
            },
            'data_browsing': {
                'name': 'waveform-visualizer',
                'description': 'Browse directories for seismic data files',
                'parameters': ['directory', 'file_type']
            },
            'waveform_plotting': {
                'name': 'waveform-visualizer',
                'description': 'Plot seismic waveform data',
                'parameters': ['file_path', 'output', 'filter', 'time_window']
            },
            'seismo_statistics': {
                'name': 'seismo-stats',
                'description': 'Compute b-value, Mc, G-R relation and seismicity distribution plots',
                'parameters': ['input_path', 'output_prefix', 'mc', 'method', 'plot']
            }
        }

    def execute(self, intent: str, entities: Dict, context: ConversationContext) -> Dict:
        """
        Execute the appropriate skill
        Returns: {success: bool, message: str, results: dict}
        """
        if intent == 'batch_picking':
            return self._execute_batch_picking(entities, context)
        elif intent == 'confirm_picking':
            return self._execute_confirm_picking(entities, context)
        elif intent == 'phase_picking':
            return self._execute_phase_picking(entities, context)
        elif intent == 'phase_association':
            return self._execute_phase_association(entities, context)
        elif intent == 'polarity_analysis':
            return self._execute_polarity_analysis(entities, context)
        elif intent == 'data_browsing':
            return self._execute_data_browsing(entities, context)
        elif intent == 'waveform_plotting':
            return self._execute_waveform_plotting(entities, context)
        elif intent == 'seismo_statistics':
            return self._execute_seismo_statistics(entities, context)
        elif intent == 'seismo_qa':
            return self._execute_seismo_qa(entities, context)
        elif intent == 'seismo_programming':
            return self._execute_seismo_programming(entities, context)
        elif intent == 'seismo_agent':
            return self._execute_seismo_agent(entities, context)
        elif intent == 'tool_documentation':
            return self._execute_tool_documentation(entities, context)
        else:
            return {
                'success': False,
                'message': f'Unknown intent: {intent}',
                'results': {}
            }

    def _detect_and_update_config(self, input_dir: str, project_root: str) -> dict:
        """Scan the input directory, auto-detect data format, update pnsn/config/picker.py.

        Returns a dict describing what was detected (for display in the message).
        """
        import collections

        ALL_KNOWN_CHANNELS = {
            'BHE', 'BHN', 'BHZ',
            'SHE', 'SHN', 'SHZ',
            'HHE', 'HHN', 'HHZ',
            'EIE', 'EIN', 'EIZ',
            'HNE', 'HNN', 'HNZ',
        }
        SUPPORTED_EXTS = {'.mseed', '.sac', '.seed', '.miniseed',
                          '.MSEED', '.SAC', '.SEED'}

        # --- Collect sample files ---
        sample_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in SUPPORTED_EXTS:
                    sample_files.append((root, f))
            if len(sample_files) >= 50:
                break

        if not sample_files:
            return {}

        # --- 1. Most common extension ---
        ext_counter = collections.Counter(
            os.path.splitext(f)[1].lower() for _, f in sample_files)
        filenametag = ext_counter.most_common(1)[0][0]  # e.g. '.sac'

        # --- 2. Detect channelindex and namekeyindex from filename structure ---
        channelindex = 3   # default
        namekeyindex = [0, 1]  # default
        detected_channels: set = set()

        for _, fname in sample_files[:20]:
            stem = fname.rsplit('.', 1)[0]           # strip extension
            parts = stem.split('.')
            for idx, part in enumerate(parts):
                if part.upper() in ALL_KNOWN_CHANNELS:
                    channelindex = idx
                    detected_channels.add(part.upper())
                    break

        # namekeyindex: first two parts that look like NET.STA (short codes)
        # We keep [0,1] as default since that's the standard format.
        # But verify by checking if splitting sample filenames gives unique NET.STA keys
        sample_stems = [fname.rsplit('.', 1)[0] for _, fname in sample_files[:20]]
        for stem in sample_stems:
            parts = stem.split('.')
            if len(parts) > channelindex:
                # Everything before channelindex that isn't a known channel looks like NET.STA.LOC
                # Typically [0] = NET, [1] = STA
                namekeyindex = [0, 1]
                break

        # --- 3. Detect sampling rate from a real file ---
        samplerate = 100  # default
        if HAS_OBSPY:
            import obspy as _obspy
            for root, fname in sample_files[:5]:
                try:
                    st = _obspy.read(os.path.join(root, fname), headonly=True)
                    if st:
                        sr = st[0].stats.sampling_rate
                        if sr > 0:
                            samplerate = int(sr)
                            break
                except Exception:
                    continue

        # --- 4. Build chnames from detected channels ---
        # Group into component sets: E/N/Z variants
        COMP_GROUPS = [
            ['BHE', 'BHN', 'BHZ'],
            ['SHE', 'SHN', 'SHZ'],
            ['HHE', 'HHN', 'HHZ'],
            ['EIE', 'EIN', 'EIZ'],
            ['HNE', 'HNN', 'HNZ'],
        ]
        # Only include groups that have at least one detected channel
        active_groups = [g for g in COMP_GROUPS
                         if any(ch in detected_channels for ch in g)]
        if not active_groups:
            active_groups = COMP_GROUPS  # fallback to all

        # --- 5. Write updated config ---
        config_path = os.path.join(project_root, 'pnsn', 'config', 'picker.py')
        chnames_str = str(active_groups)
        config_content = f'''\


class Parameter:
    # 数据设置 — 由 SeismicX AI 自动检测于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    nchannel = 3                 # 通道数量
    samplerate = {samplerate}             # 采样率 (自动检测)

    # 拾取设置，仅对onnx模型有用
    prob = 0.3
    nmslen = 1000
    npicker = 1
    npre = 2

    is_seed = True
    filenametag = "{filenametag}"        # 文件扩展名 (自动检测)
    # 文件名格式: {sample_files[0][1] if sample_files else "未知"}
    namekeyindex = {namekeyindex}        # NET.KEY索引 (自动检测)
    channelindex = {channelindex}              # 分量标识索引 (自动检测)

    chnames = {chnames_str}
    polar = True
    ifplot = False
    ifreal = False
    snritv = 100
    bandpass = [1, 10]


#par = Parameter()
'''
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        return {
            'filenametag': filenametag,
            'samplerate': samplerate,
            'channelindex': channelindex,
            'namekeyindex': namekeyindex,
            'detected_channels': sorted(detected_channels),
            'active_groups': active_groups,
            'sample_file': sample_files[0][1] if sample_files else '',
        }

    def _resolve_input_dir(self, entities: Dict, context: ConversationContext):
        """从 entities 或上下文中解析输入目录。"""
        if 'file_paths' in entities:
            return entities['file_paths'][0]
        if context.last_results.get('browse_directory'):
            return context.last_results['browse_directory']
        if context.last_results.get('browse_files'):
            return str(Path(context.last_results['browse_files'][0]).parent)
        # 最后兜底：从最近的对话历史中扫描路径字符串
        path_pattern = r'(?:\.{1,2})?/[A-Za-z0-9_\-./~:]+'
        for msg in reversed(context.conversation_history):
            if msg['role'] == 'user':
                raw_paths = re.findall(path_pattern, msg['content'])
                cleaned = [p.rstrip('.,;:!?，。；：！？、)】」』"\'') for p in raw_paths]
                cleaned = [p for p in cleaned if p]
                if cleaned:
                    return cleaned[0]
                break
        return None

    def _execute_batch_picking(self, entities: Dict, context: ConversationContext) -> Dict:
        """第一步：扫描目录，若有不足三分量台站则询问用户处理方式。"""
        SagePicker = _get_sage_picker_class()
        if SagePicker is None:
            return {'success': False,
                    'message': '无法加载 SagePicker 模块，请确保 pnsn/sage_picker.py 存在。',
                    'results': {}}

        input_dir = self._resolve_input_dir(entities, context)
        if not input_dir or not os.path.exists(input_dir):
            return {'success': False,
                    'message': '请告诉我要遍历的波形目录路径。',
                    'needs_info': ['input_directory'], 'results': {}}

        project_root = str(Path(__file__).parent)
        model_rel = entities.get('model', 'pnsn/pickers/pnsn.v3.jit')
        model_path = os.path.join(project_root, model_rel)
        if not os.path.exists(model_path):
            return {'success': False,
                    'message': f'模型文件不存在: {model_path}', 'results': {}}

        # 扫描目录
        try:
            picker = SagePicker(model_path, samplerate=100.0)
            scan = picker.scan_directory(input_dir)
        except Exception as e:
            return {'success': False, 'message': f'扫描目录失败: {e}', 'results': {}}

        n_complete   = scan['n_complete']
        n_incomplete = scan['n_incomplete']
        incomplete_keys = scan['incomplete_keys']

        # 保存 pending 任务到上下文
        context.current_task = 'batch_picking_pending'
        context.task_state = {
            'input_dir': input_dir,
            'model_path': model_path,
            'project_root': project_root,
            'incomplete_keys': incomplete_keys,
        }

        if n_incomplete == 0:
            # 全部完整，直接开始
            return self._run_sage_picker(input_dir, model_path, 'skip', context)

        # 有不足三分量台站，询问用户
        incomplete_list = '\n'.join(f'  - {k}' for k in incomplete_keys[:10])
        if n_incomplete > 10:
            incomplete_list += f'\n  ... 共 {n_incomplete} 个'

        return {
            'success': True,
            'message': (f'📋 目录扫描完成\n\n'
                        f'  完整三分量台站: **{n_complete}** 个\n'
                        f'  不足三分量台站: **{n_incomplete}** 个\n\n'
                        f'不足三分量台站列表:\n{incomplete_list}\n\n'
                        f'请问不足三分量的台站怎么处理？\n'
                        f'  • **跳过** — 直接跳过，只处理完整台站\n'
                        f'  • **复制** — 将现有分量复制成三份（仍可拾取，精度略低）'),
            'action': 'need_user_input',
            'results': {}
        }

    def _execute_confirm_picking(self, entities: Dict, context: ConversationContext) -> Dict:
        """第二步：用户回答跳过/复制后，实际启动拾取。"""
        if context.current_task != 'batch_picking_pending':
            return {'success': False,
                    'message': '当前没有待确认的批量拾取任务。请先说"遍历目录拾取所有震相"。',
                    'results': {}}

        state = context.task_state
        input_dir  = state.get('input_dir', '')
        model_path = state.get('model_path', '')

        # 判断用户选择
        # entities['raw_text'] 不一定有，从 context.conversation_history 取最后一条 user 消息
        last_user = ''
        for msg in reversed(context.conversation_history):
            if msg['role'] == 'user':
                last_user = msg['content'].lower()
                break

        if any(w in last_user for w in ['复制', 'duplicate', '补齐', '三份']):
            mode = 'duplicate'
        else:
            mode = 'skip'

        return self._run_sage_picker(input_dir, model_path, mode, context)

    def _run_sage_picker(self, input_dir: str, model_path: str,
                         incomplete: str, context: ConversationContext) -> Dict:
        """返回 sage_picking_async action，让 app.py 在后台线程运行拾取并实时报告进度。"""
        project_root = str(Path(__file__).parent)
        os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
        output_base = os.path.join(
            project_root, f'results/sage_picks_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        mode_label = '复制分量补齐' if incomplete == 'duplicate' else '跳过不足分量台站'

        context.current_task = None
        context.task_state = {}

        return {
            'success': True,
            'message': (f'⏳ 批量拾取已启动（{mode_label}），正在后台遍历目录...\n'
                        f'目录: `{input_dir}`\n'
                        f'进度会每隔几秒自动更新，完成后显示汇总。'),
            'action': 'sage_picking_async',
            'results': {
                'input_dir': input_dir,
                'model_path': model_path,
                'incomplete': incomplete,
                'output_base': output_base,
                'picks_file': f'{output_base}.txt',
            }
        }

    def _execute_phase_picking(self, entities: Dict, context: ConversationContext) -> Dict:
        """Pick phases by running the JIT model directly in-process.

        Flow:
        1. Read all waveform files with ObsPy, filter by NET.STA if specified.
        2. Group traces by NET.STA.LOC → each group needs E/N/Z components.
        3. Resample to 100 Hz, align start times, stack as [N, 3].
        4. Run torch.jit model → picks tensor [n_picks, 3] (type, sample, conf).
        5. Return waveform_data + picks_data for immediate inline rendering.
        """
        if not HAS_OBSPY:
            return {'success': False,
                    'message': '未安装 ObsPy，请运行 pip install obspy。', 'results': {}}

        try:
            import torch
        except ImportError:
            return {'success': False,
                    'message': '未安装 PyTorch，请运行 pip install torch。', 'results': {}}

        # --- 1. Resolve source directory ---
        input_dir = None
        if 'file_paths' in entities:
            input_dir = entities['file_paths'][0]
        elif context.last_results.get('browse_directory'):
            input_dir = context.last_results['browse_directory']
        elif context.last_results.get('browse_files'):
            input_dir = str(Path(context.last_results['browse_files'][0]).parent)

        # 兜底：从最近的对话历史中扫描路径
        if not input_dir:
            _path_re = r'(?:\.{1,2})?/[A-Za-z0-9_\-./~:]+'
            for _msg in reversed(context.conversation_history):
                if _msg['role'] == 'user':
                    _raw = re.findall(_path_re, _msg['content'])
                    _cleaned = [p.rstrip('.,;:!?，。；：！？、)】」』"\'') for p in _raw if p]
                    if _cleaned:
                        input_dir = _cleaned[0]
                    break

        if not input_dir or not os.path.exists(input_dir):
            return {'success': False,
                    'message': '请提供包含波形文件的目录路径。',
                    'needs_info': ['input_directory'], 'results': {}}

        # ★ 关键：若给定的是目录而非文件，自动路由到批量拾取
        if os.path.isdir(input_dir):
            return self._execute_batch_picking(entities, context)

        # --- 2. Read all waveform files ---
        supported_ext = {'.mseed', '.sac', '.seed', '.miniseed',
                         '.MSEED', '.SAC', '.SEED'}
        all_files = [p for p in Path(input_dir).rglob('*')
                     if p.suffix in supported_ext and p.stat().st_size > 0]
        if not all_files:
            return {'success': False,
                    'message': f'目录 {input_dir} 中未找到波形文件。', 'results': {}}

        import obspy as _obspy
        st_all = _obspy.Stream()
        for f in all_files:
            try:
                st_all += _obspy.read(str(f))
            except Exception:
                pass

        if len(st_all) == 0:
            return {'success': False, 'message': '所有文件读取失败，请检查格式。', 'results': {}}

        # --- 3. Filter by NET.STA ---
        target_stations = entities.get('stations', [])
        if target_stations:
            matched = _obspy.Stream()
            for tr in st_all:
                net_sta = f'{tr.stats.network}.{tr.stats.station}'.upper()
                if any(net_sta == s or tr.stats.station.upper() == s.split('.')[-1]
                       for s in target_stations):
                    matched += tr
            if len(matched) == 0:
                found = sorted({f'{t.stats.network}.{t.stats.station}' for t in st_all})
                return {'success': False,
                        'message': (f'未找到台站 {target_stations}。\n'
                                    f'目录中存在: {", ".join(found)}'), 'results': {}}
            st_filtered = matched
            station_label = ', '.join(target_stations)
        else:
            st_filtered = st_all
            station_label = '所有台站'

        # --- 4. Load JIT model ---
        project_root = str(Path(__file__).parent)
        model_rel = entities.get('model', 'pnsn/pickers/pnsn.v3.jit')
        model_path = os.path.join(project_root, model_rel)
        if not os.path.exists(model_path):
            return {'success': False,
                    'message': f'模型文件不存在: {model_path}', 'results': {}}

        device = torch.device('cpu')
        sess = torch.jit.load(model_path, map_location=device)
        sess.eval()

        # --- 5. Group by NET.STA.LOC, align & pick ---
        SAMPLE_RATE = 100.0
        PHASE_MAP = {0: 'Pg', 1: 'Sg', 2: 'Pn', 3: 'Sn', 4: 'P', 5: 'S'}
        # Component priority order for E/N/Z
        COMP_ORDER = [
            ['BHE', 'HHE', 'SHE', 'EIE', 'HNE'],
            ['BHN', 'HHN', 'SHN', 'EIN', 'HNN'],
            ['BHZ', 'HHZ', 'SHZ', 'EIZ', 'HNZ'],
        ]

        # Group traces by NET.STA.LOC
        groups: Dict[str, list] = {}
        for tr in st_filtered:
            key = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}'
            groups.setdefault(key, []).append(tr)

        all_picks = []
        waveform_data = []

        for grp_key, traces in groups.items():
            # Resample all traces to SAMPLE_RATE
            st_grp = _obspy.Stream(traces)
            for tr in st_grp:
                if tr.stats.sampling_rate != SAMPLE_RATE:
                    tr.resample(SAMPLE_RATE)

            # Find E, N, Z components
            def pick_component(grp_stream, candidates):
                for ch in candidates:
                    sel = grp_stream.select(channel=ch)
                    if sel:
                        return sel[0]
                return None

            tr_e = pick_component(st_grp, COMP_ORDER[0])
            tr_n = pick_component(st_grp, COMP_ORDER[1])
            tr_z = pick_component(st_grp, COMP_ORDER[2])

            if not (tr_e and tr_n and tr_z):
                # Fallback: just use whatever 3 traces are available
                avail = list(st_grp)
                if len(avail) < 3:
                    # Can't pick without 3 components, add waveform only
                    waveform_data.extend(
                        self._extract_waveform_data_from_stream(st_grp))
                    continue
                tr_e, tr_n, tr_z = avail[0], avail[1], avail[2]

            # Align to common start time (latest start)
            t_start = max(tr_e.stats.starttime,
                          tr_n.stats.starttime,
                          tr_z.stats.starttime)
            t_end   = min(tr_e.stats.endtime,
                          tr_n.stats.endtime,
                          tr_z.stats.endtime)
            if t_end <= t_start:
                waveform_data.extend(
                    self._extract_waveform_data_from_stream(st_grp))
                continue

            for tr in [tr_e, tr_n, tr_z]:
                tr.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)

            # Stack as [N, 3]
            n = min(len(tr_e.data), len(tr_n.data), len(tr_z.data))
            data_np = np.stack(
                [tr_e.data[:n], tr_n.data[:n], tr_z.data[:n]], axis=1
            ).astype(np.float32)

            # Run model
            with torch.no_grad():
                x = torch.tensor(data_np, dtype=torch.float32, device=device)
                y = sess(x)
                phase_arr = y.cpu().numpy()  # [n_picks, 3]: [type, sample, conf]

            # Convert picks to time in seconds
            sr = SAMPLE_RATE
            for row in phase_arr:
                phase_type = int(row[0])
                sample_idx = float(row[1])
                confidence = float(row[2])
                time_s = sample_idx / sr
                all_picks.append({
                    'phase': PHASE_MAP.get(phase_type, f'P{phase_type}'),
                    'time_s': round(time_s, 3),
                    'confidence': round(confidence, 3),
                    'station': grp_key,
                    'sample': int(sample_idx),
                })

            # Build waveform display data (use Z channel as primary)
            for tr in [tr_e, tr_n, tr_z]:
                waveform_data.extend(
                    self._extract_waveform_data_from_stream(_obspy.Stream([tr])))

        stations_found = list(groups.keys())
        n_picks = len(all_picks)

        # Store picks in context
        context.last_results['last_picks'] = all_picks

        pick_summary = ''
        if all_picks:
            for p in sorted(all_picks, key=lambda x: x['time_s'])[:10]:
                pick_summary += f"  {p['phase']} @ {p['time_s']:.2f}s  置信度:{p['confidence']:.2f}\n"
            if n_picks > 10:
                pick_summary += f'  ... 共 {n_picks} 个'
        else:
            pick_summary = '  未检测到震相'

        return {
            'success': True,
            'message': (f'✓ 拾取完成！\n\n'
                        f'台站: {station_label}\n'
                        f'台站组: {", ".join(stations_found)}\n'
                        f'检测到 {n_picks} 个震相:\n{pick_summary}'),
            'action': 'display_plot',
            'results': {
                'waveform_data': waveform_data,
                'picks_data': all_picks,
                'title': f'{station_label} — 震相拾取结果',
                'source_file': input_dir,
            }
        }

    def _extract_waveform_data_from_stream(self, st) -> list:
        """Extract display-ready waveform data from an ObsPy Stream."""
        MAX_POINTS = 4000
        traces = []
        for tr in st:
            data = tr.data.astype(float)
            sr = tr.stats.sampling_rate
            n = len(data)
            max_val = float(np.max(np.abs(data))) if n > 0 else 1.0
            if max_val > 0:
                data = data / max_val
            if n > MAX_POINTS:
                step = n // MAX_POINTS
                data = data[::step]
                times = (np.arange(len(data)) * step / sr).tolist()
            else:
                times = (np.arange(n) / sr).tolist()
            traces.append({
                'label': f'{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}',
                'times': times,
                'amplitudes': data.tolist(),
                'start_time': str(tr.stats.starttime),
                'sampling_rate': sr,
            })
        return traces

    def _execute_phase_association(self, entities: Dict, context: ConversationContext) -> Dict:
        """Execute phase association skill"""
        # Try to use previous pick results if available
        if 'input_file' not in entities:
            if context.last_results.get('picks_file'):
                input_file = context.last_results['picks_file']
            else:
                return {
                    'success': False,
                    'message': '我需要震相拾取文件的路径。您刚完成了拾取吗？或者请提供拾取文件路径。',
                    'needs_info': ['input_file', 'station_file'],
                    'results': {}
                }
        else:
            input_file = entities['input_file']

        if 'station_file' not in entities:
            return {
                'success': False,
                'message': '我需要台站文件的路径。请提供包含台站信息的文件路径。',
                'needs_info': ['station_file'],
                'results': {}
            }

        station_file = entities['station_file']
        output = entities.get('output', f'results/events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        method = entities.get('method', 'fastlink')

        cmd = f"python pnsn/{method}er.py -i {input_file} -o {output} -s {station_file}"

        context.current_task = 'phase_association'
        context.task_state = {
            'input_file': input_file,
            'station_file': station_file,
            'output': output,
            'method': method,
            'command': cmd
        }

        return {
            'success': True,
            'message': f'开始震相关联...\n拾取文件: {input_file}\n台站文件: {station_file}\n方法: {method}',
            'action': 'execute_command',
            'command': cmd,
            'results': {'output_file': output}
        }

    def _execute_polarity_analysis(self, entities: Dict, context: ConversationContext) -> Dict:
        """Execute polarity analysis skill"""
        if 'input_file' not in entities:
            if context.last_results.get('picks_file'):
                input_file = context.last_results['picks_file']
            else:
                return {
                    'success': False,
                    'message': '我需要震相拾取文件路径。',
                    'needs_info': ['input_file', 'waveform_dir'],
                    'results': {}
                }
        else:
            input_file = entities['input_file']

        if 'waveform_dir' in entities:
            waveform_dir = entities['waveform_dir']
        elif 'file_paths' in entities:
            waveform_dir = entities['file_paths'][0]
        elif context.last_results.get('browse_directory'):
            waveform_dir = context.last_results['browse_directory']
        elif context.last_results.get('browse_files'):
            waveform_dir = str(Path(context.last_results['browse_files'][0]).parent)
        else:
            return {
                'success': False,
                'message': '我需要波形文件目录路径。',
                'needs_info': ['waveform_dir'],
                'results': {}
            }

        output = f'results/polarity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        cmd = f"python seismic_cli.py polarity -i {input_file} -w {waveform_dir} -o {output}"

        return {
            'success': True,
            'message': f'开始初动极性分析...\n拾取文件: {input_file}\n波形目录: {waveform_dir}',
            'action': 'execute_command',
            'command': cmd,
            'results': {'output_file': output}
        }

    def _execute_data_browsing(self, entities: Dict, context: ConversationContext) -> Dict:
        """Execute data browsing - scan directory for seismic data files"""
        if 'file_paths' not in entities:
            return {
                'success': False,
                'message': '请提供要浏览的目录路径。例如："查看 /path/to/data 目录下的文件"',
                'needs_info': ['directory'],
                'results': {}
            }

        directory = entities['file_paths'][0]

        # Check if directory exists
        if not os.path.exists(directory):
            return {
                'success': False,
                'message': f'目录不存在: {directory}\n请检查路径是否正确。',
                'results': {}
            }

        if not os.path.isdir(directory):
            return {
                'success': False,
                'message': f'路径不是目录: {directory}\n请提供目录路径而不是文件路径。',
                'results': {}
            }

        # Scan for seismic data files
        supported_extensions = ['.mseed', '.sac', '.seed', '.miniseed']
        data_files = []

        try:
            for ext in supported_extensions:
                for file_path in Path(directory).rglob(f'*{ext}'):
                    data_files.append(file_path)

            # Also check uppercase extensions
            for ext in ['.MSEED', '.SAC', '.SEED']:
                for file_path in Path(directory).rglob(f'*{ext}'):
                    data_files.append(file_path)

            data_files = sorted(set(data_files))  # Remove duplicates and sort

        except Exception as e:
            return {
                'success': False,
                'message': f'扫描目录时出错: {e}',
                'results': {}
            }

        if not data_files:
            return {
                'success': True,
                'message': f'在目录 {directory} 中未找到地震数据文件。\n支持的格式: {", ".join(supported_extensions)}',
                'results': {'files': [], 'count': 0}
            }

        # Format file list for display
        file_list = []
        for i, file_path in enumerate(data_files[:20], 1):  # Limit to first 20 files
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_list.append(f"{i}. {file_path.name} ({size_mb:.1f} MB)")

        if len(data_files) > 20:
            file_list.append(f"... 还有 {len(data_files) - 20} 个文件")

        message = f"在目录 {directory} 中找到 {len(data_files)} 个地震数据文件:\n\n"
        message += "\n".join(file_list)
        message += f"\n\n您可以对我说 '绘制第1个文件' 或 '绘制 {data_files[0].name}' 来查看波形。"

        # Store results in context for follow-up
        context.last_results['browse_directory'] = directory
        context.last_results['browse_files'] = [str(f) for f in data_files]

        return {
            'success': True,
            'message': message,
            'action': 'display_files',
            'results': {
                'files': [str(f) for f in data_files],
                'count': len(data_files),
                'directory': directory
            }
        }

    def _execute_waveform_plotting(self, entities: Dict, context: ConversationContext) -> Dict:
        """Execute waveform plotting using ObsPy and matplotlib"""
        if not HAS_OBSPY:
            return {
                'success': False,
                'message': '未安装 ObsPy 库。请运行 "pip install obspy" 来安装。',
                'results': {}
            }

        if not HAS_MATPLOTLIB:
            return {
                'success': False,
                'message': '未安装 matplotlib 库。请运行 "pip install matplotlib" 来安装。',
                'results': {}
            }

        # --- Collect all requested file paths ---
        file_paths = []
        browse_files = context.last_results.get('browse_files', [])

        if 'numbers' in entities and browse_files:
            # User said e.g. "绘制 4、5、6" — collect ALL indices
            for n in entities['numbers']:
                idx = int(n) - 1
                if 0 <= idx < len(browse_files):
                    file_paths.append(browse_files[idx])

        if not file_paths and 'file_paths' in entities:
            file_paths = entities['file_paths']

        if not file_paths and context.last_results.get('last_plotted_file'):
            file_paths = [context.last_results['last_plotted_file']]

        if not file_paths:
            return {
                'success': False,
                'message': '请指定要绘制的文件。例如：\n- "绘制 /path/to/file.mseed"\n- 或者先使用 "查看 /path/to/data" 浏览文件',
                'needs_info': ['file_path'],
                'results': {}
            }

        # Validate & expand directories
        resolved = []
        supported_extensions = ['.mseed', '.sac', '.seed', '.miniseed',
                                 '.MSEED', '.SAC', '.SEED']
        for fp in file_paths:
            if not os.path.exists(fp):
                return {'success': False, 'message': f'文件不存在: {fp}', 'results': {}}
            if os.path.isdir(fp):
                candidates = sorted([
                    p for ext in supported_extensions
                    for p in Path(fp).rglob(f'*{ext}')
                    if p.stat().st_size > 0
                ])
                if not candidates:
                    return {'success': False,
                            'message': f'目录 {fp} 中未找到波形文件。',
                            'results': {}}
                resolved.append(str(candidates[0]))
            else:
                resolved.append(fp)

        # Extract waveform data from all files and merge traces
        try:
            all_traces = []
            file_names = []
            for fp in resolved:
                traces = self._extract_waveform_data(fp)
                all_traces.extend(traces)
                file_names.append(Path(fp).name)

            start_time = all_traces[0]['start_time'] if all_traces else ''
            title = ' | '.join(file_names)

            return {
                'success': True,
                'message': f'✓ 波形图已生成！\n\n文件: {chr(10).join(file_names)}\n开始时间: {start_time}',
                'action': 'display_plot',
                'results': {
                    'waveform_data': all_traces,
                    'title': title,
                    'source_file': resolved[0]
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'绘制波形时出错: {e}\n\n请确保文件格式正确且可读。',
                'results': {}
            }

    def _extract_waveform_data(self, file_path: str) -> list:
        """Extract waveform data as JSON-serializable list for web rendering."""
        st = obspy.read(file_path)
        if len(st) == 0:
            raise ValueError("文件中没有数据道")

        MAX_POINTS = 4000  # 每道最多传输的采样点数，超出则降采样
        traces = []
        for tr in st:
            data = tr.data.astype(float)
            sr = tr.stats.sampling_rate
            n = len(data)

            # Normalise
            max_val = float(np.max(np.abs(data))) if n > 0 else 1.0
            if max_val > 0:
                data = data / max_val

            # Downsample if necessary
            if n > MAX_POINTS:
                step = n // MAX_POINTS
                data = data[::step]
                times = (np.arange(len(data)) * step / sr).tolist()
            else:
                times = (np.arange(n) / sr).tolist()

            traces.append({
                'label': f'{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}',
                'times': times,
                'amplitudes': data.tolist(),
                'start_time': str(tr.stats.starttime),
                'sampling_rate': sr,
            })
        return traces

    def _plot_waveform_internal(self, file_path: str, output_image: str, entities: Dict) -> str:
        """Internal method to plot waveform using ObsPy and matplotlib"""
        # Read the data
        st = obspy.read(file_path)

        if len(st) == 0:
            raise ValueError("文件中没有数据道")

        # Create figure
        n_traces = len(st)
        fig_height = max(3, n_traces * 2.5)
        fig, axes = plt.subplots(n_traces, 1, figsize=(12, fig_height), sharex=True)

        if n_traces == 1:
            axes = [axes]

        # Normalize traces for better visualization
        for i, tr in enumerate(st):
            data = tr.data
            if len(data) > 0:
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val  # Normalize to [-1, 1]

            # Plot
            times = np.arange(len(tr.data)) / tr.stats.sampling_rate
            axes[i].plot(times, data, 'b-', linewidth=0.8)
            axes[i].set_ylabel(f'{tr.stats.network}.{tr.stats.station}\n{tr.stats.channel}',
                              fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Set x-axis label
        axes[-1].set_xlabel('Time (seconds)', fontsize=10)

        # Title
        start_time = st[0].stats.starttime
        fig.suptitle(f'Seismic Waveform\n{Path(file_path).name}\nStart: {start_time.strftime("%Y-%m-%d %H:%M:%S")}',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_image

    # ------------------------------------------------------------------
    # Seismological Statistics  (b-value, Mc, G-R, distribution plots)
    # ------------------------------------------------------------------

    def _execute_seismo_statistics(self, entities: Dict, context: ConversationContext) -> Dict:
        """
        Compute b-value, Mc and generate analysis plots from a catalog or picks file.

        Handles three cases:
        1. User explicitly supplies a file/directory path
        2. Agent uses the last picks result stored in context
        3. Agent searches the results/ directory for the newest picks file
        """
        import sys
        from pathlib import Path as _Path

        project_root = str(_Path(__file__).parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # --- Resolve input path ---
        input_path = None

        # Priority 1: explicit path from user message
        if entities.get('file_paths'):
            input_path = entities['file_paths'][0]

        # Priority 2: last picks file stored in context
        if not input_path and context.last_results.get('picks_file'):
            input_path = context.last_results['picks_file']

        # Priority 3: newest sage_picks_*.txt in results/
        if not input_path:
            results_dir = _Path(project_root) / 'results'
            picks_files = sorted(results_dir.glob('sage_picks_*.txt')) if results_dir.exists() else []
            if picks_files:
                input_path = str(picks_files[-1])

        if not input_path:
            return {
                'success': False,
                'message': (
                    '请告诉我震相结果文件或地震目录文件的路径，例如：\n'
                    '"计算 /path/to/results/ 的b值"\n'
                    '或者先运行震相检测，再进行统计分析。'
                ),
                'needs_info': ['catalog_path'],
                'results': {}
            }

        # --- Load catalog / picks ---
        try:
            from seismo_stats.catalog_loader import load_picks_txt, load_catalog_file
            p = _Path(input_path)
            if p.is_dir() or (p.is_file() and 'sage_picks' in p.name):
                catalog = load_picks_txt(input_path)
                data_type = 'picks'
            else:
                catalog = load_catalog_file(input_path)
                data_type = 'catalog'
        except Exception as e:
            return {
                'success': False,
                'message': f'无法加载数据文件：{e}\n请检查路径是否正确。',
                'results': {}
            }

        # --- Determine what to compute ---
        user_text = ' '.join([
            m['content'] for m in context.conversation_history[-3:]
            if m['role'] == 'user'
        ]) if context.conversation_history else ''

        want_bvalue   = any(k in user_text for k in ['b值', 'b-value', 'bvalue', 'gutenberg', 'richter', '统计'])
        want_mc       = any(k in user_text for k in ['Mc', 'mc', '完整性', 'completeness']) or want_bvalue
        want_temporal = any(k in user_text for k in ['时间', '时序', 'temporal', '时空'])
        want_spatial  = any(k in user_text for k in ['空间', '位置', '分布', 'spatial', 'map', '地图'])

        # Default: compute b-value + all available plots
        if not any([want_bvalue, want_temporal, want_spatial]):
            want_bvalue = True

        # --- Output prefix ---
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = _Path(project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        output_prefix = str(results_dir / f'stats_{ts}')

        # --- Collect magnitudes (picks file doesn't have magnitudes) ---
        messages = []
        result_bv = None
        plots = {}

        if want_bvalue or want_mc:
            if not catalog.has_magnitudes:
                messages.append(
                    '⚠️  当前数据来自震相拾取文件，不含震级信息，无法计算b值。\n'
                    '请提供包含震级列（magnitude/mag/ML等）的地震目录文件（CSV/JSON格式）来计算b值。'
                )
                want_bvalue = False
            else:
                try:
                    from seismo_stats.bvalue import calc_bvalue_mle
                    result_bv = calc_bvalue_mle(catalog.magnitudes, mc_method='maxcurvature')
                    messages.append(result_bv.summary())
                except Exception as e:
                    messages.append(f'b值计算失败：{e}')

        # --- Plots ---
        try:
            from seismo_stats.plotting import plot_gr, plot_temporal, plot_spatial

            if result_bv is not None:
                try:
                    p_gr = plot_gr(result_bv, output_prefix + '_gr.png')
                    plots['gr'] = p_gr
                except Exception as e:
                    messages.append(f'G-R图绘制失败：{e}')

            if (want_temporal or want_bvalue) and catalog.times:
                try:
                    p_t = plot_temporal(catalog, output_prefix + '_temporal.png')
                    plots['temporal'] = p_t
                except Exception as e:
                    messages.append(f'时间分布图绘制失败：{e}')

            if want_spatial and catalog.has_locations:
                try:
                    p_s = plot_spatial(catalog, output_prefix + '_spatial.png')
                    plots['spatial'] = p_s
                except Exception as e:
                    messages.append(f'空间分布图绘制失败：{e}')
            elif want_spatial and not catalog.has_locations:
                messages.append('⚠️  数据中没有经纬度信息，无法绘制空间分布图。')

        except ImportError as e:
            messages.append(f'绘图模块未安装：{e}（请 pip install matplotlib）')

        # --- Build summary message ---
        summary_lines = [f'📊 地震统计分析完成  （数据：{_Path(input_path).name}）']
        summary_lines.append(f'   共 {len(catalog)} 条记录')
        if catalog.times:
            summary_lines.append(
                f'   时间范围：{min(catalog.times).strftime("%Y-%m-%d %H:%M")}  →  '
                f'{max(catalog.times).strftime("%Y-%m-%d %H:%M")}'
            )
        if messages:
            summary_lines.append('')
            summary_lines.extend(messages)

        if plots:
            summary_lines.append('\n已生成图像：')
            labels = {'gr': 'G-R频率震级关系图', 'temporal': '时间分布图', 'spatial': '空间分布图'}
            for k, path in plots.items():
                summary_lines.append(f'  • {labels.get(k, k)}: {path}')

        # Store results in context
        context.last_results.update({
            'stats_output_prefix': output_prefix,
            'stats_plots': plots,
            'stats_bvalue': result_bv.b_value if result_bv else None,
            'stats_mc': result_bv.mc if result_bv else None,
        })

        return {
            'success': True,
            'message': '\n'.join(summary_lines),
            'action': 'display_plot' if plots else 'none',
            'results': {
                'stats_plots': plots,
                'bvalue_result': {
                    'b': result_bv.b_value,
                    'b_unc': result_bv.b_uncertainty,
                    'a': result_bv.a_value,
                    'mc': result_bv.mc,
                    'n': result_bv.n_events,
                } if result_bv else None,
            }
        }

    # ------------------------------------------------------------------
    # Seismo Programming  (LLM 代码生成 + 执行)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Seismo QA  (纯对话问答，不执行代码)
    # ------------------------------------------------------------------

    # ── QA 系统提示（专业地震学助手）────────────────────────────────────────
    _QA_SYSTEM_PROMPT = (
        "你是 SAGE 系统内置的地震学专家助手，具备深厚的地震学和地震数据处理知识。\n"
        "回答用户问题时请遵循以下原则：\n"
        "1. 用准确、简洁的中文回答，专业术语可附英文原文\n"
        "2. 涉及数据处理时，优先给出基于 ObsPy/Python 的代码示例（```python 包裹）\n"
        "3. 如果问题涉及 SAGE 内置工具（seismo_code.toolkit），优先展示工具包用法\n"
        "4. 代码示例后可简短提示：说「帮我对 /data/xxx.mseed 做XXX」可以让我直接执行\n"
        "5. 不要重复用户的问题，直接给出答案\n"
        "\n"
        "SAGE 内置工具包（seismo_code.toolkit）主要函数：\n"
        "- read_stream(path)           — 读取波形文件或目录\n"
        "- filter_stream(st, type, freqmin, freqmax)  — 滤波（bandpass/lowpass/highpass）\n"
        "- plot_stream(st, title, outfile)             — 绘制波形\n"
        "- remove_response(st, inventory_path)         — 去仪器响应\n"
        "- taup_arrivals(dist_deg, depth_km)           — 计算理论走时\n"
        "- compute_hvsr(st, freqmin, freqmax)          — HVSR 谱比\n"
        "- estimate_magnitude_ml(tr, dist_km)          — ML 震级估算\n"
    )

    def _call_llm_for_qa(
        self,
        messages: list,
        llm_config: Dict,
        timeout: int = 30,
    ) -> Optional[str]:
        """向 LLM 发送对话请求，返回回复文本，失败返回 None。"""
        provider = llm_config.get('provider', 'ollama')
        model    = llm_config.get('model', '')
        api_base = llm_config.get('api_base', '')
        api_key  = llm_config.get('api_key', '')

        if not api_base or not model:
            return None

        if provider == 'ollama':
            url     = api_base.rstrip('/') + '/api/chat'
            payload = {'model': model, 'messages': messages, 'stream': False,
                       'options': {'temperature': 0.6, 'num_predict': 2000}}
        else:
            url     = api_base.rstrip('/') + '/chat/completions'
            payload = {'model': model, 'messages': messages,
                       'temperature': 0.6, 'max_tokens': 2000}

        try:
            data = json.dumps(payload).encode()
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}' if api_key else 'Bearer none',
            }
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())

            if provider == 'ollama':
                return body.get('message', {}).get('content', '').strip() or None
            else:
                return (body.get('choices', [{}])[0]
                        .get('message', {}).get('content', '').strip()) or None
        except Exception:
            return None

    def _execute_seismo_qa(self, entities: Dict, context: ConversationContext) -> Dict:
        """
        用 LLM 回答地震学知识问题。
        LLM 不可用时给出诚实说明并提示配置后端，不再返回硬编码帮助菜单。
        """
        import sys as _sys
        from pathlib import Path as _Path
        project_root = str(_Path(__file__).parent)
        if project_root not in _sys.path:
            _sys.path.insert(0, project_root)

        # 取当前用户问题
        user_msgs = [m['content'] for m in context.conversation_history if m['role'] == 'user']
        question = user_msgs[-1] if user_msgs else ''

        # 获取 LLM 配置（直接读磁盘 config.json，与 LLM 设置页保持同步）
        llm_config: Optional[Dict] = None
        try:
            from config_manager import LLMConfigManager
            llm_config = LLMConfigManager().get_llm_config()
        except Exception:
            pass

        if llm_config and llm_config.get('model') and llm_config.get('api_base'):
            # 构建带历史上下文的消息列表
            messages: list = [{'role': 'system', 'content': self._QA_SYSTEM_PROMPT}]
            # 最近 6 条历史（3 轮）
            for msg in context.conversation_history[-6:]:
                if msg['role'] in ('user', 'assistant'):
                    messages.append({'role': msg['role'], 'content': msg['content']})

            answer = self._call_llm_for_qa(messages, llm_config, timeout=30)
            if answer:
                return {'success': True, 'message': answer, 'action': 'none', 'results': {}}

        # ---- LLM 真的不可用 ----
        backend_hint = (
            "当前没有配置可用的 LLM 模型，无法回答此问题。\n\n"
            "请前往 **[LLM 设置](/llm-settings)** 页面，选择已安装的 Ollama 模型并保存，"
            "然后再次提问即可。\n\n"
            f"> 你的问题：「{question}」"
        )
        return {'success': True, 'message': backend_hint, 'action': 'none', 'results': {}}

    @staticmethod
    def _builtin_seismo_qa(question: str) -> str:
        """保留接口兼容性，实际已由 _execute_seismo_qa 的 LLM 路径替代。"""
        return (
            "当前没有可用的 LLM 后端。请运行 `sage backend setup` 配置后端后重试。"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # （原内置知识库已移除，全部交由 LLM 动态回答）
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_seismo_agent(self, entities: Dict, context: ConversationContext) -> Dict:
        """触发地震学 Agent：阅读文献 → 自主规划 → 逐步编程实现。"""
        import sys
        from pathlib import Path as _Path
        project_root = str(_Path(__file__).parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        user_msgs = [m['content'] for m in context.conversation_history if m['role'] == 'user']
        user_text = user_msgs[-1] if user_msgs else ''

        try:
            from seismo_agent.agent_loop import SeismoAgent
        except ImportError:
            return {
                'success': False,
                'message': (
                    'seismo_agent 模块尚未完全加载。\n'
                    '请确认 seismo_agent/ 目录已存在。\n'
                    '您可以通过命令行使用：\n'
                    '  python seismic_cli.py agent --paper /path/to/paper.pdf "实现其中的滤波方法"'
                ),
                'results': {}
            }

        # Resolve paper source from entities or text
        paper_source = None
        for fp in entities.get('file_paths', []):
            if fp.endswith('.pdf') or 'arxiv' in fp.lower() or fp.startswith('10.'):
                paper_source = fp
                break
        if not paper_source:
            # Look for arXiv IDs or DOIs in text
            import re as _re
            ax = _re.search(r'\d{4}\.\d{4,}', user_text)
            doi = _re.search(r'10\.\d{4,}/\S+', user_text)
            if ax:
                paper_source = ax.group()
            elif doi:
                paper_source = doi.group()

        llm_config = None
        try:
            from config_manager import LLMConfigManager
            llm_config = LLMConfigManager().get_llm_config()
        except Exception:
            pass

        agent = SeismoAgent(llm_config=llm_config, project_root=project_root)

        results_dir = _Path(project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = str(results_dir / f'agent_{ts}')

        # Run with callback for streaming updates
        lines = []
        def _cb(msg: str):
            lines.append(msg)

        final = agent.run(
            goal=user_text,
            paper_source=paper_source,
            output_dir=output_dir,
            progress_cb=_cb,
        )

        context.last_results['agent_output_dir'] = output_dir
        context.last_results['agent_figures'] = final.get('figures', [])

        return {
            'success': final.get('success', False),
            'message': final.get('summary', '\n'.join(lines)),
            'action': 'display_plot' if final.get('figures') else 'none',
            'results': final,
        }

    # ------------------------------------------------------------------
    # Seismo Programming  (LLM 代码生成 + 执行)
    # ------------------------------------------------------------------

    def _execute_seismo_programming(self, entities: Dict, context: ConversationContext) -> Dict:
        """
        接受自然语言地震学编程需求，LLM 生成并执行 Python 代码。
        """
        import sys
        from pathlib import Path as _Path

        project_root = str(_Path(__file__).parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Gather user request from conversation history
        user_msgs = [m['content'] for m in context.conversation_history
                     if m['role'] == 'user']
        user_request = user_msgs[-1] if user_msgs else ''

        # Resolve data hint (most recently mentioned file path)
        data_hint = None
        if entities.get('file_paths'):
            data_hint = entities['file_paths'][0]
        elif context.last_results.get('last_plotted_file'):
            data_hint = context.last_results['last_plotted_file']
        elif context.last_results.get('browse_files'):
            files = context.last_results['browse_files']
            if files:
                data_hint = files[0]

        # Check if LLM is available
        try:
            from seismo_code.code_engine import get_code_engine
        except ImportError as e:
            return {
                'success': False,
                'message': f'seismo_code 模块导入失败: {e}',
                'results': {}
            }

        engine = get_code_engine()

        if not engine.is_llm_available():
            # LLM unavailable — return helpful message with toolkit reference
            return {
                'success': False,
                'message': (
                    '⚠️  LLM 服务不可用（Ollama 未运行或 API Key 未配置）。\n\n'
                    '代码生成功能需要 LLM 支持。您可以：\n'
                    '  1. 启动 Ollama: `ollama serve` 后重试\n'
                    '  2. 或配置在线 API: `python seismic_cli.py llm setup`\n\n'
                    '您也可以直接使用内置地震学工具包编写代码：\n'
                    '  from seismo_code.toolkit import read_stream, filter_stream, plot_stream\n'
                    '  st = read_stream("/path/to/data.mseed")\n'
                    '  st = filter_stream(st, "bandpass", freqmin=1, freqmax=10)\n'
                    '  plot_stream(st)'
                ),
                'results': {}
            }

        # Run code generation + execution
        try:
            result = engine.run(user_request, data_hint=data_hint, timeout=90)
        except Exception as e:
            return {
                'success': False,
                'message': f'代码引擎出错: {e}',
                'results': {}
            }

        # Copy generated figures to results/
        results_dir = _Path(project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_figures = []
        for i, fig_path in enumerate(result.figures):
            import shutil
            src = _Path(fig_path)
            if src.exists():
                dst = results_dir / f'prog_{ts}_{i}{src.suffix}'
                shutil.copy2(str(src), str(dst))
                saved_figures.append(str(dst))

        # Build response message
        lines = []
        if result.success:
            lines.append('✅ 代码执行成功')
        else:
            lines.append('⚠️  代码执行遇到问题')

        if result.stdout.strip():
            output_text = result.stdout.strip()
            # Remove [FIGURE] lines from display output
            display_lines = [l for l in output_text.splitlines() if not l.startswith('[FIGURE]')]
            if display_lines:
                lines.append('\n输出结果：')
                lines.extend(['  ' + l for l in display_lines])

        if saved_figures:
            lines.append(f'\n生成图像 ({len(saved_figures)} 张)：')
            for f in saved_figures:
                lines.append(f'  • {f}')

        if not result.success and result.exec_result:
            err = result.exec_result.error or result.exec_result.stderr
            if err:
                lines.append(f'\n错误信息：{err[:300]}')

        # Store in context
        context.last_results['prog_figures'] = saved_figures
        context.last_results['prog_code'] = result.code

        return {
            'success': result.success,
            'message': '\n'.join(lines),
            'action': 'display_plot' if saved_figures else 'none',
            'results': {
                'figures': saved_figures,
                'code': result.code,
                'stdout': result.stdout,
            }
        }

    # ------------------------------------------------------------------
    # Tool Documentation  (解析文档 + 工具调用)
    # ------------------------------------------------------------------

    def _execute_tool_documentation(self, entities: Dict, context: ConversationContext) -> Dict:
        """
        解析外部工具文档，注册到工具库，并支持调用已知工具。
        """
        import sys
        from pathlib import Path as _Path

        project_root = str(_Path(__file__).parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        user_msgs = [m['content'] for m in context.conversation_history
                     if m['role'] == 'user']
        user_text = user_msgs[-1] if user_msgs else ''

        try:
            from seismo_tools.tool_registry import list_tools, get_tool, run_tool, generate_input_files
        except ImportError as e:
            return {'success': False, 'message': f'seismo_tools 模块导入失败: {e}', 'results': {}}

        # --- Subcase 1: list available tools ---
        if any(k in user_text for k in ['有哪些工具', '支持什么工具', '工具列表', 'list tools']):
            tools = list_tools()
            builtin_names = ['HypoDD', 'VELEST', 'NonLinLoc', 'HYPOINVERSE', 'FOCMEC']
            lines = ['📋 已注册的地震学外部工具：\n']
            lines.append('内置工具（已有完整接口模板）：')
            for t in builtin_names:
                p = get_tool(t.lower())
                if p:
                    lines.append(f'  • {t}: {p.get("description", "")[:60]}...')
            user_tools = [t for t in tools if t not in [b.lower() for b in builtin_names]]
            if user_tools:
                lines.append('\n用户注册工具：')
                for t in user_tools:
                    lines.append(f'  • {t}')
            lines.append('\n使用方式：直接告诉我"用HypoDD重定位"或"解析这个工具文档"')
            return {'success': True, 'message': '\n'.join(lines), 'action': 'none', 'results': {}}

        # --- Subcase 2: query known tool info ---
        known_tools = ['hypodd', 'velest', 'nonlinloc', 'hypoinverse', 'focmec']
        for tool_key in known_tools:
            if tool_key in user_text.lower():
                profile = get_tool(tool_key)
                if profile:
                    lines = [f'🔧 {profile["name"]} 工具说明\n']
                    lines.append(f'功能: {profile["description"]}')
                    lines.append(f'可执行程序: {profile["executable"]}')
                    lines.append(f'输入文件: {", ".join(profile["input_files"])}')
                    lines.append(f'输出文件: {", ".join(profile["output_files"])}')
                    lines.append(f'\n输入格式说明:\n{profile["input_format"][:500]}')
                    if profile.get("run_command"):
                        lines.append(f'\n调用命令: {profile["run_command"]}')
                    if profile.get("notes"):
                        lines.append(f'\n注意: {profile["notes"]}')
                    lines.append('\n💡 要生成输入文件，请说明您的数据情况，例如：')
                    lines.append(f'  "用{profile["name"]}重定位，震相文件在 results/sage_picks_xxx.txt，台站文件在 data/stations.txt"')
                    return {'success': True, 'message': '\n'.join(lines), 'action': 'none', 'results': {}}

        # --- Subcase 3: parse provided documentation (text or file path) ---
        doc_text = None
        doc_file = None

        # Check for file path in entities
        for fpath in entities.get('file_paths', []):
            p = _Path(fpath)
            if p.exists() and p.suffix.lower() in ('.txt', '.md', '.rst', '.pdf'):
                doc_file = str(p)
                break

        # Check for file path in stored context
        if not doc_file and context.last_results.get('last_doc_file'):
            doc_file = context.last_results['last_doc_file']

        # If a long user message looks like documentation (>200 chars), treat as doc
        if not doc_file and len(user_text) > 200:
            doc_text = user_text

        if not doc_file and not doc_text:
            return {
                'success': False,
                'message': (
                    '请告诉我工具文档的路径，或直接粘贴工具的 README / 使用说明，例如：\n'
                    '  "解析文档 /path/to/tool_readme.txt"\n'
                    '  "有哪些可用工具？"\n'
                    '  或直接粘贴工具说明文本'
                ),
                'needs_info': ['tool_documentation'],
                'results': {}
            }

        try:
            from seismo_code.doc_parser import DocParser
        except ImportError as e:
            return {'success': False, 'message': f'文档解析模块导入失败: {e}', 'results': {}}

        llm_config = None
        try:
            from config_manager import LLMConfigManager
            llm_config = LLMConfigManager().get_llm_config()
        except Exception:
            pass

        parser = DocParser(llm_config)

        try:
            if doc_file:
                profile = parser.parse_file(doc_file)
                context.last_results['last_doc_file'] = doc_file
            else:
                profile = parser.parse_text(doc_text)
        except Exception as e:
            return {'success': False, 'message': f'文档解析失败: {e}', 'results': {}}

        saved_path = profile.save()
        context.last_results['last_tool_profile'] = profile.name

        lines = [
            f'✅ 已解析并注册工具：{profile.name}\n',
            profile.summary(),
            f'\n已保存到: {saved_path}',
            '\n现在您可以说：',
            f'  "用 {profile.name} 处理数据"',
            '  "生成输入文件"',
        ]
        return {
            'success': True,
            'message': '\n'.join(lines),
            'action': 'none',
            'results': {'tool_name': profile.name, 'profile_path': saved_path}
        }


class ResponseGenerator:
    """Generates natural language responses"""

    def __init__(self):
        self.templates = {
            'greeting': [
                "您好！我是SeismicX智能助手。我可以帮助您进行地震数据分析，包括震相检测、震相关联、初动极性分析和波形可视化。请问有什么可以帮您的？",
                "欢迎使用SeismicX！我可以帮您处理地震数据。您可以告诉我想要做什么，比如'查看目录'、'绘制波形'、'拾取震相'或'关联地震事件'。",
            ],
            'help': [
                "我可以帮您完成以下任务：\n\n1. **数据浏览** - 查看目录中的地震数据文件\n   例如：'帮我查看 /data/waveforms 目录下有哪些mseed文件'\n\n2. **波形绘制** - 绘制地震波形图\n   例如：'绘制 /path/to/file.mseed 的波形'\n\n3. **震相检测** - 从波形文件中检测Pg/Sg/Pn/Sn震相\n   例如：'帮我拾取/data/waveforms目录下的震相'\n\n4. **震相关联** - 将震相拾取关联为地震事件\n   例如：'关联刚才的拾取结果'\n\n5. **初动极性分析** - 分析P波初动方向\n   例如：'分析极性'\n\n请告诉我您想做什么？",
            ],
            'confirmation': [
                "好的，我将执行：{action}\n\n参数：\n{params}\n\n是否继续？(是/否)",
            ],
            'missing_info': [
                "我需要更多信息：\n{missing}\n\n请提供这些信息。",
            ],
            'success': [
                "✓ 完成！{result}\n\n接下来您想做什么？",
            ],
            'error': [
                "✗ 出错了：{error}\n\n请检查参数后重试，或者告诉我详细信息以便我帮助您。",
            ]
        }

    def generate(self, response_type: str, **kwargs) -> str:
        """Generate response based on type"""
        templates = self.templates.get(response_type, [""])
        template = templates[0]  # Could randomize later

        try:
            return template.format(**kwargs)
        except KeyError:
            return template


class ConversationalAgent:
    """Main conversational agent that orchestrates the dialogue"""

    def __init__(self):
        self.context = ConversationContext()
        self.intent_classifier = IntentClassifier()
        self.skill_executor = SkillExecutor()
        self.response_generator = ResponseGenerator()
        self.config_manager = None
        self.llm_agent = None
        self.llm_error: Optional[str] = None

        # Try to import config manager
        try:
            from config_manager import get_config_manager
            self.config_manager = get_config_manager()
        except ImportError:
            pass

        # Try to build an LLM-backed tool-use agent. If it's not available
        # (Ollama not running, import error, model doesn't support tools),
        # we silently fall back to the rule-based path.
        try:
            from llm_agent import build_agent_from_config
            self.llm_agent = build_agent_from_config(
                self.skill_executor, self.context
            )
        except Exception as e:  # noqa: BLE001 — any failure means "no LLM"
            self.llm_error = str(e)
            self.llm_agent = None

    def process_message(self, user_message: str) -> Dict:
        """
        Process a user message and generate response
        Returns: {response: str, action: str, data: dict}
        """
        # Add to conversation history (used by both paths)
        self.context.add_message('user', user_message)

        # ---- LLM path (preferred) ---------------------------------
        if self.llm_agent is not None:
            try:
                result = self.llm_agent.process_message(user_message)
                self.context.add_message('assistant', result.get('response', ''))
                return result
            except Exception as e:  # noqa: BLE001
                # 降级：连续两次出错就永久切回规则引擎，避免一直卡
                self.llm_error = f"{type(e).__name__}: {e}"
                self.llm_agent = None
                # Fall through to rule-based path below

        # ---- Rule-based fallback ----------------------------------
        # Classify intent
        intent_result = self.intent_classifier.classify(user_message)

        # Handle based on intent
        if intent_result['intent'] == 'help':
            response = self.response_generator.generate('help')
            return {
                'response': response,
                'action': 'none',
                'data': {}
            }

        elif intent_result['intent'] == 'unknown':
            response = "抱歉，我没有理解您的意思。您可以问我'能做什么'来获取帮助，或者直接描述您想完成的任务。"
            return {
                'response': response,
                'action': 'none',
                'data': {}
            }

        # Execute skill
        execution_result = self.skill_executor.execute(
            intent_result['intent'],
            intent_result['entities'],
            self.context
        )

        # Generate response
        if not execution_result['success']:
            if 'needs_info' in execution_result:
                missing = '\n'.join([f"- {info}" for info in execution_result['needs_info']])
                response = self.response_generator.generate('missing_info', missing=missing)
                return {
                    'response': response,
                    'action': 'request_info',
                    'data': execution_result
                }
            else:
                response = self.response_generator.generate('error', error=execution_result['message'])
                return {
                    'response': response,
                    'action': 'error',
                    'data': execution_result
                }

        # Success - store results
        if 'results' in execution_result:
            self.context.last_results.update(execution_result['results'])

        response = execution_result['message']
        self.context.add_message('assistant', response)

        return {
            'response': response,
            'action': execution_result.get('action', 'none'),
            'data': execution_result
        }

    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.context.conversation_history

    def reset_conversation(self):
        """Reset conversation context"""
        self.context = ConversationContext()


# Singleton instance
_agent_instance = None


def get_agent() -> ConversationalAgent:
    """Get or create agent singleton"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance


if __name__ == '__main__':
    # Interactive mode
    print("=" * 80)
    print("SeismicX Conversational Agent")
    print("=" * 80)
    print()

    agent = get_agent()

    # Greeting
    greeting = agent.response_generator.generate('greeting')
    print(f"Assistant: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'bye']:
                print("\nAssistant: 再见！祝您工作顺利！👋\n")
                break

            if not user_input:
                continue

            result = agent.process_message(user_input)
            print(f"\nAssistant: {result['response']}\n")

            # If there's an action to execute
            if result['action'] == 'execute_command' and 'command' in result['data']:
                confirm = input("执行命令？(y/n): ").strip().lower()
                if confirm == 'y':
                    import subprocess
                    cmd = result['data']['command']
                    print(f"\nExecuting: {cmd}\n")
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if proc.returncode == 0:
                        print("✓ 命令执行成功！")
                    else:
                        print(f"✗ 命令执行失败:\n{proc.stderr}")
                    print()

        except KeyboardInterrupt:
            print("\n\nAssistant: 再见！👋\n")
            break
        except Exception as e:
            print(f"\nAssistant: 发生错误: {e}\n请重试。\n")
