"""技能和工作流管理路由

v2 新增接口：
  POST /api/skills/install          从本地目录安装文件夹技能
  GET  /api/skills/<name>/references 获取技能的 references 内容
  GET  /api/skills/folder-template  获取文件夹技能的初始模板文件集
"""
from flask import Blueprint, request, jsonify
from helpers import get_skill_loader, get_workflow_runner

bp = Blueprint('skills', __name__)


# ── Skills API ──────────────────────────────────────────────────────────────

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
    # references 内容可能很大；默认只返回 ref_names，加 ?include_refs=1 才返回全文
    result = {**detail}
    if request.args.get('include_refs') != '1':
        result['references'] = list(detail.get('references', {}).keys())
    return jsonify({'ok': True, **result})


@bp.route('/api/skills/<name>/references', methods=['GET'])
def skills_references(name):
    """
    获取指定技能的全部 references 内容。

    可选参数：
      ?ref=<ref_name>  只返回指定 reference 文件的内容
    """
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    detail = sl.get_skill_detail(name)
    if detail is None:
        return jsonify({'ok': False, 'error': f'未找到技能：{name}'}), 404

    refs = detail.get('references', {})
    if not refs:
        return jsonify({'ok': True, 'references': {}, 'ref_names': []})

    ref_filter = request.args.get('ref', '').strip()
    if ref_filter:
        if ref_filter not in refs:
            return jsonify({'ok': False, 'error': f'未找到 reference：{ref_filter}'}), 404
        return jsonify({'ok': True, 'ref_name': ref_filter, 'content': refs[ref_filter]})

    return jsonify({'ok': True, 'references': refs, 'ref_names': list(refs.keys())})


@bp.route('/api/skills', methods=['POST'])
def skills_save():
    """新建或更新用户自定义单文件技能。"""
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


@bp.route('/api/skills/install', methods=['POST'])
def skills_install():
    """
    从本地目录安装文件夹技能到 ~/.seismicx/skills/<name>/。

    请求体（JSON）：
      {
        "path": "/absolute/or/relative/path/to/skill-folder",
        "overwrite": true   // 可选，默认 true
      }

    成功返回：
      { "ok": true, "name": "skill-name", "path": "...", "ref_names": [...] }
    """
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500

    data      = request.json or {}
    src_path  = (data.get('path') or '').strip()
    overwrite = data.get('overwrite', True)

    if not src_path:
        return jsonify({'ok': False, 'error': '请提供 path 字段（技能源目录路径）'}), 400

    try:
        entry = sl.skill_loader.install_skill_from_dir(src_path, overwrite=overwrite)
        return jsonify({
            'ok':       True,
            'name':     entry['name'],
            'path':     entry['path'],
            'is_folder': entry.get('is_folder', True),
            'ref_names': list(entry.get('references', {}).keys()),
            'agent_config': entry.get('agent_config', {}),
        })
    except FileNotFoundError as e:
        return jsonify({'ok': False, 'error': str(e)}), 404
    except FileExistsError as e:
        return jsonify({'ok': False, 'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'安装失败：{e}'}), 500


@bp.route('/api/skills/<name>', methods=['DELETE'])
def skills_delete(name):
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    ok = sl.skill_loader.delete_user_skill(name)
    if not ok:
        return jsonify({
            'ok': False,
            'error': f'未找到可删除的用户技能：{name}（内置技能不可删除）'
        }), 404
    return jsonify({'ok': True})


@bp.route('/api/skills/template', methods=['GET'])
def skills_template():
    """返回新技能的 Markdown 模板（单文件格式）。"""
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'template': ''})
    name     = request.args.get('name', 'my_skill')
    title    = request.args.get('title', '我的技能')
    keywords = request.args.get('keywords', '关键词1, 关键词2')
    desc     = request.args.get('desc', '功能描述')
    tpl = sl.skill_loader.SKILL_TEMPLATE.format(
        name=name, title=title, keywords=keywords, description=desc
    )
    return jsonify({'ok': True, 'template': tpl})


@bp.route('/api/skills/folder-template', methods=['GET'])
def skills_folder_template():
    """
    返回文件夹技能的初始文件集模板。

    返回 {"ok": true, "files": {"SKILL.md": "...", "agents/openai.yaml": "...", ...}}
    """
    sl = get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'files': {}})

    name         = request.args.get('name', 'my_skill')
    title        = request.args.get('title', '我的技能')
    desc         = request.args.get('desc', '功能描述')
    display_name = request.args.get('display_name', title)

    templates = sl.skill_loader.FOLDER_SKILL_TEMPLATE
    files = {}
    for filename, tpl in templates.items():
        files[filename] = tpl.format(
            name=name,
            title=title,
            description=desc,
            display_name=display_name,
        )

    return jsonify({'ok': True, 'files': files})


# ── Workflow API ────────────────────────────────────────────────────────────

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
        return jsonify({
            'ok': False,
            'error': f'未找到可删除的用户工作流：{name}（内置工作流不可删除）'
        }), 404
    return jsonify({'ok': True})


@bp.route('/api/workflows/template', methods=['GET'])
def workflows_template():
    """返回新工作流 Markdown + YAML frontmatter 模板。"""
    name       = request.args.get('name',     'my_workflow')
    title      = request.args.get('title',    '我的工作流')
    keywords   = request.args.get('keywords', '关键词1, 关键词2')
    skills_str = request.args.get('skills',   '')
    desc       = request.args.get('desc',     '工作流描述')

    kw_list   = [k.strip() for k in keywords.split(',') if k.strip()]
    kw_items  = '\n'.join(f'  - {k}' for k in kw_list) if kw_list else '  - keyword'

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
