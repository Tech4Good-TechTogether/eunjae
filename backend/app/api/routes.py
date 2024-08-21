from flask import Blueprint, render_template, Response, request, jsonify
from flask_cors import cross_origin
from .hand_landmark import gen
from .pose_analysis import analyze_frame

api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@api_bp.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@api_bp.route('/analyze_frame', methods=['POST', 'OPTIONS'])
def analyze_frame_route():
    if request.method == 'OPTIONS':
        return '', 204  # OPTIONS 요청에 대해 204 상태 코드를 반환

    data = request.json
    image = data.get('image')
    result = analyze_frame(image)
    return jsonify(result)