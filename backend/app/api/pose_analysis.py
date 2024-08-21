import cv2
import mediapipe as mp
import numpy as np
import base64
import pickle
from io import BytesIO
from PIL import Image
from dtw import accelerated_dtw
import os
from scipy.spatial.distance import euclidean

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

hand_landmarks_dict = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP"
}

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# static/landmarks 폴더의 절대 경로를 계산
landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks.pkl')

# 가이드 영상의 좌표를 미리 로드
with open(landmarks_path, 'rb') as f:
    guide_landmarks = pickle.load(f)

def compare_joint(series_a, series_b):
    # frame_num, joint_num, 3(x,y,z)
    sim = 0
    for joint_a, joint_b in zip(series_a, series_b):
        dist, _, _, _ = accelerated_dtw(joint_a, joint_b, dist="matching")
        sim += dist
    sim /= len(series_a)
    return sim

def compare_hand(guide_landmarks, result_landmarks):
    guide_landmarks = np.array(guide_landmarks)
    result_landmarks = np.array(result_landmarks)
    
    dist_joints = []
    max_joint = [None,None]
    
    for i in range(2):
        guide_hand = guide_landmarks[:,i,:,:]
        result_hand = result_landmarks[:,i,:,:]
        hand_joints = []
        
        for joint in range(21):
            series_a = guide_hand[:,joint,:]
            series_b = result_hand[:,joint,:]
            dist = compare_joint(series_a, series_b)
            hand_joints.append(dist)
            
        dist_joints.append(hand_joints)
        if max(hand_joints)>0.1:
            max_idx=np.argmax(hand_joints)
            max_joint[i] = hand_landmarks_dict[max_idx]
    
    return dist_joints, max_joint
    
def analyze_frames(video_data):    
    try:
        landmarks = []
        for image_data in video_data:
            # base64로 인코딩된 이미지를 디코딩
            image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Mediapipe로 이미지 처리
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.append([lm.x, lm.y, lm.z])
                    landmarks.append(np.array(hand_data))
        
        guide_landmarks = np.array(guide_landmarks)
        landmarks = np.array(landmarks)
        
        # 가이드 영상과 현재 영상의 손동작 비교
        if len(landmarks) > 0:
            dist_joints, _ = compare_hand(guide_landmarks, landmarks)
            
            comparison_result = {
                "distance": dist_joints,
                #"best_match_frame": best_match_frame.tolist() if best_match_frame is not None else []
            }
        else:
            comparison_result = {"distance": None, "best_match_frame": []}

        return {
            "status": "Hand detected",
            "landmarks": True,
            "results": [lm.tolist() for lm in landmarks],  # numpy 배열을 리스트로 변환하여 반환
            "comparison": comparison_result  # DTW 결과 추가
        }
        
        if len(landmarks)==0:
            return {
                "status": "No hand detected",
                "landmarks": False,
                "results": [],
                "comparison": None  # 손이 감지되지 않으면 비교 결과도 None
            }

    except Exception as e:
        return {
            "status": "Error",
            "message": str(e)
        }


def analyze_frame(image_data):
    try:
        # base64로 인코딩된 이미지를 디코딩
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Mediapipe로 이미지 처리
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append([lm.x, lm.y, lm.z])
                landmarks.append(np.array(hand_data))

            # 가이드 영상과 현재 영상의 손동작 비교
            if len(landmarks) > 0:
                current_frame_landmarks = landmarks[0]  # 첫 번째 손만 비교
                min_dist = float('inf')
                best_match_frame = None

                for guide_frame_landmarks in guide_landmarks:
                    if len(guide_frame_landmarks) > 0:
                        guide_frame_landmarks = guide_frame_landmarks[0]
                        dist, _, _, _ = accelerated_dtw(current_frame_landmarks, guide_frame_landmarks, dist=euclidean)
                        if dist < min_dist:
                            min_dist = dist
                            best_match_frame = guide_frame_landmarks
                
                comparison_result = {
                    "distance": min_dist,
                    "best_match_frame": best_match_frame.tolist() if best_match_frame is not None else []
                }
            else:
                comparison_result = {"distance": None, "best_match_frame": []}

            return {
                "status": "Hand detected",
                "landmarks": True,
                "results": [lm.tolist() for lm in landmarks],  # numpy 배열을 리스트로 변환하여 반환
                "comparison": comparison_result  # DTW 결과 추가
            }
        else:
            return {
                "status": "No hand detected",
                "landmarks": False,
                "results": [],
                "comparison": None  # 손이 감지되지 않으면 비교 결과도 None
            }

    except Exception as e:
        return {
            "status": "Error",
            "message": str(e)
        }
