"""routes — 注册所有 Flask Blueprint"""
from .pages      import bp as pages_bp
from .llm        import bp as llm_bp
from .workspace  import bp as workspace_bp
from .skills     import bp as skills_bp
from .knowledge  import bp as knowledge_bp
from .code       import bp as code_bp
from .chat       import bp as chat_bp


def register_blueprints(app):
    app.register_blueprint(pages_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(workspace_bp)
    app.register_blueprint(skills_bp)
    app.register_blueprint(knowledge_bp)
    app.register_blueprint(code_bp)
    app.register_blueprint(chat_bp)
