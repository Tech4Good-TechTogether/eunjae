import cv2
import mediapipe as mp
import pickle
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# static/landmarks 폴더의 절대 경로를 계산
landmarks_path = os.path.join(current_dir,  'landmarks', 'guide_landmarks.pkl')

# 가이드 영상의 좌표를 미리 로드
with open(landmarks_path, 'rb') as f:
    guide_landmarks = pickle.load(f)

# 가이드 랜드마크 중 첫 번째 프레임만 사용 (예시)
guide_frame_landmarks = guide_landmarks[0][0]

# 웹캠에서 프레임 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 프레임을 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 스켈레톤 그리기
    for lm in guide_frame_landmarks:
        h, w, c = frame.shape
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Mediapipe의 HAND_CONNECTIONS를 사용하여 연결선 그리기
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = guide_frame_landmarks[start_idx]
        end_point = guide_frame_landmarks[end_idx]

        start_x, start_y = int(start_point[0] * w), int(start_point[1] * h)
        end_x, end_y = int(end_point[0] * w), int(end_point[1] * h)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # 결과를 화면에 보여주기
    cv2.imshow('Webcam with Guide Hand Skeleton', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
