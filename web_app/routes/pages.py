"""页面路由：返回 HTML 模板"""
from flask import Blueprint, render_template, redirect, url_for, jsonify
from datetime import datetime
from state import tasks

bp = Blueprint('pages', __name__)


@bp.route('/')
def index():
    return redirect(url_for('pages.chat_page'))


@bp.route('/chat')
def chat_page():
    return render_template('chat.html')


@bp.route('/knowledge')
def knowledge_page():
    return render_template('knowledge.html')


@bp.route('/skills')
def skills_page():
    return render_template('skills.html')


@bp.route('/llm-settings')
def llm_settings_page():
    return render_template('llm_settings.html')


@bp.route('/evidence-geo-agent')
def evidence_geo_agent_page():
    return render_template('evidence_geo.html')


@bp.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_tasks': sum(1 for t in tasks.values() if t.get('status') == 'running'),
    })
