"""代码执行和工作流路由"""
from flask import Blueprint, request, jsonify
import os
import sys
import threading
import time as _time
import uuid as _uuid
from pathlib import Path
from state import _code_engine_lock, _code_engines, _code_jobs, _PROJECT_ROOT
from helpers import (
    get_llm_config, get_code_engine, gc_code_jobs, serialize_code_result,
    get_user_profile_context,
)

bp = Blueprint('code', __name__)


@bp.route('/api/chat/code', methods=['POST'])
def chat_code():
    """
    Start an async code-generation job.
    Returns immediately with {ok, job_id}.
    Frontend polls /api/chat/code/poll/<job_id> for progress + result.
    """
    data       = request.json or {}
    user_msg   = (data.get('message') or '').strip()
    session_id = data.get('session_id', 'default')
    if not user_msg:
        return jsonify({'ok': False, 'error': '消息不能为空'}), 400

    llm_cfg = get_llm_config()
    if not llm_cfg.get('api_base'):
        job_id = _uuid.uuid4().hex[:8]
        _code_jobs[job_id] = {
            'status': 'done', 'progress': [], 'ts': _time.time(),
            'result': {'ok': True, 'success': False,
                       'response': '未配置 LLM 后端，请在 LLM 设置页面完成配置。',
                       'code': '', 'stdout': '', 'figures': [],
                       'skill_used': None, 'attempts': 0,
                       'debug_trace': [], 'plan': [],
                       'script_b64': '', 'downloads': []},
        }
        return jsonify({'ok': True, 'job_id': job_id})

    gc_code_jobs()
    job_id = _uuid.uuid4().hex[:8]
    _code_jobs[job_id] = {'status': 'running', 'progress': [], 'result': None,
                          'cancelled': False, 'ts': _time.time()}

    def _run():
        try:
            from config_manager import get_config_manager
            coding_cfg = get_config_manager().get_coding_agent_config()
            coding_backend = (coding_cfg.get('backend') or 'builtin').strip()
            use_coding_backend = coding_backend in {'aider', 'openhands'}

            # Phase 1: Init under lock — protects sentence-transformers / C-extension loading
            with _code_engine_lock:
                from seismo_skill import search_skills, invalidate_cache
                invalidate_cache()
                try:
                    hits = search_skills(user_msg, top_k=1)
                    skill_used = hits[0]['name'] if hits else None
                except Exception:
                    skill_used = None

                engine = None if use_coding_backend else get_code_engine(session_id, llm_cfg)

            # Phase 2: Execute OUTSIDE lock — subprocess (GMT/Python) is fork-safe;
            # holding the lock here would block all other init for minutes.
            def _on_progress(p):
                if isinstance(p, dict):
                    _code_jobs[job_id]['progress'].append(p.get('message') or p.get('phase', ''))
                else:
                    _code_jobs[job_id]['progress'].append(str(p))

            profile = get_user_profile_context(max_chars=2500)
            engine_msg = user_msg
            if profile:
                engine_msg += (
                    "\n\n===== Long-term user profile (soft context; do not mention unless useful) =====\n"
                    + profile
                )

            if use_coding_backend:
                from seismo_code.external_coding_agent import run_external_coding_agent
                result = run_external_coding_agent(
                    engine_msg,
                    workspace=str(_PROJECT_ROOT),
                    llm_config=llm_cfg,
                    config=coding_cfg,
                    progress_cb=_on_progress,
                )
            else:
                result = engine.run(
                    engine_msg,
                    timeout=180,
                    max_debug_rounds=6,
                    run_verify=False,
                    on_progress=_on_progress,
                )
            if not _code_jobs[job_id].get('cancelled'):
                _code_jobs[job_id]['result'] = serialize_code_result(result, skill_used)
        except Exception as exc:
            if not _code_jobs[job_id].get('cancelled'):
                _code_jobs[job_id]['result'] = {'ok': False, 'error': str(exc)}
        finally:
            if _code_jobs[job_id].get('cancelled'):
                _code_jobs[job_id]['status'] = 'cancelled'
                _code_jobs[job_id]['result'] = {
                    'ok': False, 'cancelled': True, 'error': '已停止当前代码任务'
                }
            else:
                _code_jobs[job_id]['status'] = 'done'

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@bp.route('/api/chat/code/poll/<job_id>', methods=['GET'])
def poll_code_job(job_id):
    """Poll progress and result for an async code job."""
    job = _code_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    return jsonify({
        'status':   job['status'],          # 'running' | 'done'
        'progress': job['progress'],        # list of progress message strings
        'result':   job['result'],          # None while running, payload when done
    })


@bp.route('/api/chat/code/cancel/<job_id>', methods=['POST'])
def cancel_code_job(job_id):
    """Mark an async code job as cancelled so the UI can stop polling it."""
    job = _code_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'job not found'}), 404
    job['cancelled'] = True
    job['status'] = 'cancelled'
    job['result'] = {'ok': False, 'cancelled': True, 'error': '已停止当前代码任务'}
    job.setdefault('progress', []).append('cancelled')
    return jsonify({'ok': True})


@bp.route('/api/chat/code/reset', methods=['POST'])
def chat_code_reset():
    """清除指定 session 的 CodeEngine 历史（对话清空时调用）。"""
    session_id = (request.json or {}).get('session_id', 'default')
    if session_id in _code_engines:
        _code_engines[session_id].reset()
    return jsonify({'ok': True})


@bp.route('/api/chat/workflow', methods=['POST'])
def chat_workflow():
    """
    Run a named workflow step-by-step via CodeEngine.run_workflow().

    Body JSON:
      workflow_name  str   — workflow name (must exist in seismo_skill/workflows/)
      message        str   — user's original request (provides extra context)
      session_id     str   — engine session (default 'default')
      data_hint      str   — optional data path forwarded to each step
      skip_on_failure bool — continue past failed steps (default false)

    Returns immediately with {ok, job_id}.
    Poll /api/chat/code/poll/<job_id> for progress + result.
    """
    data          = request.json or {}
    workflow_name = (data.get('workflow_name') or '').strip()
    user_msg      = (data.get('message') or '').strip()
    session_id    = data.get('session_id', 'default')
    data_hint     = (data.get('data_hint') or '').strip() or None
    skip_on_fail  = bool(data.get('skip_on_failure', False))

    if not workflow_name:
        return jsonify({'ok': False, 'error': 'workflow_name 不能为空'}), 400

    llm_cfg = get_llm_config()
    if not llm_cfg.get('api_base'):
        job_id = _uuid.uuid4().hex[:8]
        _code_jobs[job_id] = {
            'status': 'done', 'progress': [], 'ts': _time.time(),
            'result': {'ok': False, 'error': '未配置 LLM 后端'},
        }
        return jsonify({'ok': True, 'job_id': job_id})

    gc_code_jobs()
    job_id = _uuid.uuid4().hex[:8]
    _code_jobs[job_id] = {'status': 'running', 'progress': [], 'result': None,
                          'cancelled': False, 'ts': _time.time()}

    def _run():
        try:
            # Acquire lock to prevent thread-safety issues with sentence-transformers
            with _code_engine_lock:
                def _on_progress(p):
                    # Phase keys: 'workflow_step', 'step_done', 'workflow_done'
                    label = p.get('message') or p.get('phase', '')
                    _code_jobs[job_id]['progress'].append(label)

                engine = get_code_engine(session_id, llm_cfg)
                profile = get_user_profile_context(max_chars=2000)
                workflow_msg = user_msg
                if profile:
                    workflow_msg += (
                        "\n\n===== Long-term user profile (soft context; do not mention unless useful) =====\n"
                        + profile
                    )
                result = engine.run_workflow(
                    workflow_name=workflow_name,
                    user_request=workflow_msg,
                    data_hint=data_hint,
                    max_debug_rounds=5,
                    timeout=180,
                    skip_on_failure=skip_on_fail,
                    on_progress=_on_progress,
                )

                # Collect all downloadable outputs
                import base64 as _b64
                downloads = []
                for fp in result.all_figures + result.all_output_files:
                    try:
                        with open(fp, 'rb') as _f:
                            downloads.append({
                                'filename': Path(fp).name,
                                'b64':      _b64.b64encode(_f.read()).decode(),
                                'mime':     'image/png' if fp.endswith('.png') else 'application/octet-stream',
                            })
                    except Exception:
                        pass

                if not _code_jobs[job_id].get('cancelled'):
                    _code_jobs[job_id]['result'] = {
                        'ok':           True,
                        'success':      result.success,
                        'response':     result.response,
                        'workflow_name': result.workflow_name,
                        'workflow_title': result.workflow_title,
                        'steps_total':  result.steps_total,
                        'steps_done':   result.steps_done,
                        'exec_dir':     result.exec_dir,
                        'step_results': [
                            {
                                'step_id':      sr.step_id,
                                'skill':        sr.skill,
                                'description':  sr.description,
                                'success':      sr.success,
                                'skipped':      sr.skipped,
                                'attempts':     sr.attempts,
                                'diagnosis':    sr.diagnosis,
                                'stdout':       sr.stdout[:800],
                                'stderr':       sr.stderr[:400],
                                'figures':      [Path(f).name for f in sr.figures],
                                'output_files': [Path(f).name for f in sr.output_files],
                            }
                            for sr in result.step_results
                        ],
                        'figures':   result.all_figures,
                        'downloads': downloads,
                    }
        except Exception as exc:
            if not _code_jobs[job_id].get('cancelled'):
                _code_jobs[job_id]['result'] = {'ok': False, 'error': str(exc)}
        finally:
            if _code_jobs[job_id].get('cancelled'):
                _code_jobs[job_id]['status'] = 'cancelled'
                _code_jobs[job_id]['result'] = {
                    'ok': False, 'cancelled': True, 'error': '已停止当前代码任务'
                }
            else:
                _code_jobs[job_id]['status'] = 'done'

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@bp.route('/api/chat/route', methods=['POST'])
def chat_route():
    """
    用 LLM 判断用户意图，返回路由类型：
      code  — 需要执行代码/技能
      qa    — 知识问答
      chat  — 普通对话
    """
    import re as _re

    data        = request.json or {}
    message     = data.get('message', '').strip()
    kb_has_docs = data.get('kb_has_docs', False)
    history     = data.get('history', [])   # [{role, content}, ...]

    if not message:
        return jsonify({'ok': True, 'intent': 'chat'})

    llm_cfg = get_llm_config()

    msg_stripped = message.strip()

    msg_lower = msg_stripped.lower()

    def _has_any(words):
        return any(word in msg_lower for word in words)

    mentions_paper = _has_any(['论文', '文献', '文章', '这篇', '该文', 'paper', 'article', 'pdf'])
    asks_to_read = _has_any(['解析', '解读', '分析一下', '读一下', '阅读', '总结', '概括', '讲解', '解释', '看一下', '梳理', '提炼', 'summarize', 'summarise', 'explain', 'review', 'interpret'])
    asks_to_code = _has_any(['写代码', '代码实现', '编程', 'python', '复现', '运行', '执行', '绘制', '画图', '生成图', '计算', '处理数据', '检测', '拾取', '导出', '保存', 'script', 'code', 'plot', 'run', 'execute', 'implement'])
    paper_read_request = (kb_has_docs or mentions_paper) and mentions_paper and asks_to_read and not asks_to_code
    if paper_read_request:
        return jsonify({'ok': True, 'intent': 'qa', 'rule': 'paper_read_guard'})

    # ── 唯一快速路径：含绝对路径且无问号 → 必然是 code，无需问 LLM ─────────
    has_path = bool(_re.search(r'(?:^|[\s\u4e00-\u9fff，。：、])[/~][\w./\-]{4,}', message))
    ends_q   = bool(_re.search(r'[?？]\s*$', msg_stripped))
    if has_path and not ends_q:
        return jsonify({'ok': True, 'intent': 'code'})

    # Deterministic QA guard: explanations of algorithms/methods should not be
    # sent to the coding engine just because they contain technical terms.
    explanation_re = _re.compile(
        r'(详细讲|讲一下|讲讲|解释|介绍|原理|机制|怎么回事|是什么|什么是|为什么|为何'
        r'|区别|优缺点|适用|局限|解析|解读|读一下|阅读|总结|概括|看一下|梳理'
        r'|how does|how .*work|what is|why |explain|tell me about|summari[sz]e|review)',
        _re.I,
    )
    concrete_code_re = _re.compile(
        r'(写代码|写个程序|代码实现|编程实现|实现一下|复现|生成脚本|运行|执行|读取|处理数据'
        r'|绘制|画图|下载|保存|导出|计算一下|检测一下|检测下|检测上面|识别一下|拾取|触发'
        r'|plot|run|execute|write.*code|implement|detect|pick|trigger)',
        _re.I,
    )
    if explanation_re.search(msg_stripped) and not concrete_code_re.search(msg_stripped):
        return jsonify({'ok': True, 'intent': 'qa', 'rule': 'explanation_guard'})

    apply_method_re = _re.compile(
        r'(使用|用|通过|基于).{0,30}(算法|方法|模型|STA/?LTA|stalta|classic_sta_lta|recursive_sta_lta)'
        r'.{0,30}(检测|识别|拾取|触发|处理|分析|计算)'
        r'|(?:检测|识别|拾取).{0,20}(波形|数据|事件|震相)',
        _re.I,
    )
    if apply_method_re.search(msg_stripped):
        return jsonify({'ok': True, 'intent': 'code', 'rule': 'apply_method_guard'})

    contextual_code_re = _re.compile(
        r'(用\s*Python|python|代码|编程|实现|复现|写出来|写个程序|完整实现|run it|code it|implement)',
        _re.I,
    )
    context_ref_re = _re.compile(r'(上述|上面|上文|前面|刚才|这个|该|此|this|above|previous)', _re.I)
    if contextual_code_re.search(msg_stripped) and (context_ref_re.search(msg_stripped) or history):
        return jsonify({'ok': True, 'intent': 'code', 'rule': 'contextual_code_guard'})

    # ── 构建对话历史摘要（最近 3 轮）────────────────────────────────────────
    history_text = ""
    if history:
        lines = []
        for h in history[-6:]:
            role_label = "用户" if h.get("role") == "user" else "AI"
            lines.append(f"{role_label}：{str(h.get('content',''))[:100]}")
        history_text = "\n".join(lines)
    profile_text = get_user_profile_context(max_chars=1200)

    # ── 精简 prompt：短而直接，适配弱模型 ────────────────────────────────────
    routing_prompt = f"""Classify the intent of the following user message. Output only one of: code, qa, or chat. Do not output anything else.

Definitions:
code = The user is asking the system to immediately perform a concrete operation or produce an executable/usable result, such as plotting, drawing, calculation, filtering, reading files, downloading data, processing data, writing code, generating scripts, or creating outputs.
qa   = The user is asking for explanation, guidance, concepts, principles, methods, troubleshooting ideas, or general knowledge.
chat = Casual conversation, greetings, emotional expression, or content unrelated to the task system.

Decision rule:
Classify by intent, not by specific tool names.

If the message asks the system to do something concrete now, such as create, generate, write, draw, plot, calculate, read, convert, process, analyze, download, run, save, export, or modify something → code.
If the message mainly asks what, why, how, whether, or asks for an explanation/reason/method without requiring immediate execution → qa.
If the message asks to read, explain, summarize, interpret, or analyze a paper/literature/PDF/article → qa, unless it explicitly asks to write or run code.
Otherwise → chat.

Examples:
"Help me draw a topographic map" → code
"Generate a station distribution figure" → code
"Calculate the b-value" → code
"Read this waveform file" → code
"Write a script to process SAC files" → code
"Convert this catalog to CSV" → code
"Analyze the uploaded waveform/catalog data and summarize the result" → code
"Analyze this paper for me" → qa
"Summarize this uploaded PDF" → qa
"How should I draw a topographic map?" → qa
"What is the b-value?" → qa
"Why does the waveform need filtering?" → qa
"Can SAC files be read directly?" → qa
"Hello" → chat

{f"Context: {history_text}" if history_text else ""}
{f"Long-term user profile: {profile_text}" if profile_text else ""}
User message: {message}
Intent:"""

    # ── 强操作信号 re（LLM 结果兜底用）────────────────────────────────────
    ACTION_SIGNAL_RE = _re.compile(
        r'(使用[^\s]{0,16}(?:绘|画|生成|计算|滤|读|处理|下载|运行|检测|识别|拾取|触发)'
        r'|用[^\s]{0,16}(?:绘|画|生成|计算|滤|读|处理|下载|检测|识别|拾取|触发)'
        r'|^(?:绘制|画[^报面版刊]|生成|计算|读取|处理|分析|滤波|下载|检测|识别|拾取))',
        _re.I | _re.MULTILINE
    )

    try:
        from helpers import llm_call
        raw = llm_call(
            [{"role": "user", "content": routing_prompt}],
            llm_cfg,
            max_tokens=10,
        ).lower().strip()

        intent_word = None
        for word in ['code', 'qa', 'chat']:
            if word in raw:
                intent_word = word
                break

        # 若 LLM 未识别 → 默认 code（操作型系统宁可多执行，不要沉默）
        if intent_word is None:
            intent_word = 'code' if not ends_q else 'qa'

        # 兜底 override：LLM 说 qa 但消息有强操作信号且无问号 → 纠正为 code
        if intent_word == 'qa' and not ends_q and ACTION_SIGNAL_RE.search(msg_stripped):
            intent_word = 'code'

        return jsonify({'ok': True, 'intent': intent_word})

    except Exception:
        # LLM 不可用 → 规则兜底
        if paper_read_request:
            return jsonify({'ok': True, 'intent': 'qa', 'rule': 'paper_read_guard', 'fallback': True})
        FALLBACK_RE = _re.compile(
            r'(绘制|画图|画[^报面版刊]|使用|执行|运行|下载|读取|处理|滤波|计算|生成图'
            r'|检测|识别|拾取|触发|plot|filter|spectrum|detect|pick|trigger|\.sac|\.mseed|\.csv)',
            _re.I
        )
        fallback = 'code' if FALLBACK_RE.search(message) and not ends_q else 'qa'
        return jsonify({'ok': True, 'intent': fallback, 'fallback': True})
