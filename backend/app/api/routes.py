from flask import Blueprint, render_template, Response, request, jsonify
from flask_cors import cross_origin
from .hand_landmark import gen
from .pose_analysis import analyze_frame
import os
import pickle

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


@api_bp.route('/get_guide_landmarks', methods=['GET'])
def get_guide_landmarks():
    try:
        # 현재 파일의 디렉토리 경로
        current_dir = os.path.dirname(__file__)

        # static/landmarks 폴더의 절대 경로를 계산
        landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks.pkl')

        # 가이드 영상의 좌표를 미리 로드
        with open(landmarks_path, 'rb') as f:
            guide_landmarks = pickle.load(f)
        
        # NumPy 배열을 리스트로 변환
        guide_landmarks_list = [[landmark.tolist() for landmark in frame] for frame in guide_landmarks]

        return jsonify({"guide_landmarks": guide_landmarks_list})
    
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500
