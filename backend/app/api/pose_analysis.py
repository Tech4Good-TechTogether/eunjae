import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def analyze_frame(image_data):
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
                hand_data.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                })
            landmarks.append(hand_data)

        return {
            "status": "Hand detected",
            "landmarks": True,
            "results": landmarks  # 손 랜드마크 데이터를 추가로 반환
        }
    else:
        return {
            "status": "No hand detected",
            "landmarks": False,
            "results": []  # 손이 감지되지 않으면 빈 리스트 반환
        }
