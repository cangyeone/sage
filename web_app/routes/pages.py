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


@bp.route('/config')
def config_page():
    return render_template('llm_settings.html')


@bp.route('/evidence-geo-agent')
def evidence_geo_agent_page():
    return redirect(url_for('pages.parameter_optimization_page'))


@bp.route('/science-analysis-agent')
def science_analysis_agent_page():
    return render_template('science_analysis.html')


@bp.route('/parameter-optimization-agent')
def parameter_optimization_page():
    return render_template('parameter_optimization.html')


@bp.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_tasks': sum(1 for t in tasks.values() if t.get('status') == 'running'),
    })
