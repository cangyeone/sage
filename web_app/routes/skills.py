"""技能和工作流管理路由"""
from flask import Blueprint, request, jsonify
from helpers import get_skill_loader, get_workflow_runner

bp = Blueprint('skills', __name__)


# ── Skills API ─────────────────────────────────────────────────────────────

@bp.route('/api/skills', methods=['GET'])
def skills_list():
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装', 'skills': []})
    sl.invalidate_cache()
    return jsonify({'ok': True, 'skills': sl.list_skills()})


@bp.route('/api/skills/<name>', methods=['GET'])
def skills_get(name):
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    detail = sl.get_skill_detail(name)
    if detail is None:
        return jsonify({'ok': False, 'error': f'未找到技能：{name}'}), 404
    return jsonify({'ok': True, **detail})


@bp.route('/api/skills', methods=['POST'])
def skills_save():
    """新建或更新用户自定义技能。"""
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    data = request.json or {}
    name = (data.get('name') or '').strip()
    text = (data.get('text') or '').strip()
    if not name or not text:
        return jsonify({'ok': False, 'error': 'name 和 text 不能为空'}), 400
    try:
        path = sl.skill_loader.save_user_skill(name, text)
        return jsonify({'ok': True, 'path': str(path)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@bp.route('/api/skills/<name>', methods=['DELETE'])
def skills_delete(name):
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    ok = sl.skill_loader.delete_user_skill(name)
    if not ok:
        return jsonify({'ok': False, 'error': f'未找到可删除的用户技能：{name}（内置技能不可删除）'}), 404
    return jsonify({'ok': True})


@bp.route('/api/skills/template', methods=['GET'])
def skills_template():
    """返回新技能的 Markdown 模板。"""
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'template': ''})
    name = request.args.get('name', 'my_skill')
    title = request.args.get('title', '我的技能')
    keywords = request.args.get('keywords', '关键词1, 关键词2')
    desc = request.args.get('desc', '功能描述')
    tpl = sl.skill_loader.SKILL_TEMPLATE.format(
        name=name, title=title, keywords=keywords, description=desc
    )

    return jsonify({'ok': True, 'template': tpl})


# ── Workflow API ───────────────────────────────────────────────────────────

@bp.route('/api/workflows', methods=['GET'])
def workflows_list():
    wr = get_workflow_runner()
    if wr is None:
        return jsonify({'ok': False, 'error': 'workflow_runner 未找到', 'workflows': []})
    wr.invalidate_cache()
    return jsonify({'ok': True, 'workflows': wr.list_workflows()})


@bp.route('/api/workflows/<name>', methods=['GET'])
def workflows_get(name):
    wr = get_workflow_runner()
    if wr is None:
        return jsonify({'ok': False, 'error': 'workflow_runner 未找到'}), 500
    wf = wr.load_workflow(name)
    if wf is None:
        return jsonify({'ok': False, 'error': f'未找到工作流：{name}'}), 404
    return jsonify({'ok': True, **wf})


@bp.route('/api/workflows', methods=['POST'])
def workflows_save():
    """新建或更新用户自定义工作流脚本。"""
    wr = get_workflow_runner()
    if wr is None:
        return jsonify({'ok': False, 'error': 'workflow_runner 未找到'}), 500
    data = request.json or {}
    name = (data.get('name') or '').strip()
    text = (data.get('text') or '').strip()
    if not name or not text:
        return jsonify({'ok': False, 'error': 'name 和 text 不能为空'}), 400
    try:
        path = wr.save_user_workflow(name, text)
        return jsonify({'ok': True, 'path': str(path)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@bp.route('/api/workflows/<name>', methods=['DELETE'])
def workflows_delete(name):
    wr = get_workflow_runner()
    if wr is None:
        return jsonify({'ok': False, 'error': 'workflow_runner 未找到'}), 500
    ok = wr.delete_user_workflow(name)
    if not ok:
        return jsonify({'ok': False, 'error': f'未找到可删除的用户工作流：{name}（内置工作流不可删除）'}), 404
    return jsonify({'ok': True})


@bp.route('/api/workflows/template', methods=['GET'])
def workflows_template():
    """
    返回新工作流 Markdown + YAML frontmatter 模板。
    """
    name       = request.args.get('name',     'my_workflow')
    title      = request.args.get('title',    '我的工作流')
    keywords   = request.args.get('keywords', '关键词1, 关键词2')
    skills_str = request.args.get('skills',   '')
    desc       = request.args.get('desc',     '工作流描述')

    kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
    kw_items = '\n'.join(f'  - {k}' for k in kw_list) if kw_list else '  - keyword'

    skill_items = ''
    if skills_str:
        for s in [x.strip() for x in skills_str.split(',') if x.strip()]:
            skill_items += f'  - name: {s}\n    role: 该技能在本工作流中的用途\n'
    else:
        skill_items = '  - name: skill_name\n    role: 该技能在本工作流中的用途\n'

    tpl = (
        f"---\n"
        f"name: {name}\n"
        f"title: {title}\n"
        f"version: \"1.0\"\n"
        f"description: {desc}\n"
        f"keywords:\n{kw_items}\n"
        f"skills:\n{skill_items}"
        f"steps:\n"
        f"  - id: step_1\n"
        f"    skill: skill_name\n"
        f"    description: 第一步：准备工作\n"
        f"  - id: step_2\n"
        f"    skill: skill_name\n"
        f"    description: 第二步：主要流程\n"
        f"    depends_on: [step_1]\n"
        f"  - id: step_3\n"
        f"    skill: skill_name\n"
        f"    description: 第三步：输出结果\n"
        f"    depends_on: [step_2]\n"
        f"---\n\n"
        f"## {title}\n\n"
        f"> 工作流说明：{desc}\n\n"
        f"---\n\n"
        f"### Step 1: 准备工作\n\n"
        f"描述此步骤需要 code engine 生成什么代码，调用哪个工具执行。\n\n"
        f"```python\n"
        f"# code engine 将在此生成 Python 脚本\n"
        f"import numpy as np\n"
        f"```\n\n"
        f"---\n\n"
        f"### Step 2: 主要流程\n\n"
        f"描述核心逻辑步骤。\n\n"
        f"```bash\n"
        f"# code engine 将在此生成 Shell/GMT 命令\n"
        f"echo \"hello\"\n"
        f"```\n\n"
        f"---\n\n"
        f"### Step 3: 输出结果\n\n"
        f"描述预期输出文件和验证方式。\n\n"
        f"---\n\n"
        f"## 注意事项\n\n"
        f"- 注意事项 1\n"
        f"- 注意事项 2\n"
    )
    return jsonify({'ok': True, 'template': tpl})
