from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # CORS 설정
    CORS(app)

    from .api import routes
    app.register_blueprint(routes.api_bp, url_prefix='/api')  # 블루프린트 등록

    return app
