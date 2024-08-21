import cv2
import mediapipe as mp
import numpy as np
import pickle

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_and_save_hand_landmarks(video_path, output_path):
    # 영상 읽기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    all_landmarks = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 프레임을 BGR에서 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 랜드마크 탐지
        results = hands.process(image_rgb)

        frame_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append([lm.x, lm.y, lm.z])
                frame_landmarks.append(np.array(hand_data))
                
                # 랜드마크를 프레임에 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 현재 프레임의 랜드마크를 저장
        all_landmarks.append(frame_landmarks)
        
        # 랜드마크가 그려진 프레임을 화면에 표시
        cv2.imshow('Hand Landmarks', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # 추출한 랜드마크를 pickle 파일로 저장
    with open(output_path, 'wb') as f:
        pickle.dump(all_landmarks, f)

    print(f"Saved landmarks to {output_path}")

# 예제 실행
video_path = 'videos/hand_guide.mp4'  # 입력할 가이드 영상 파일 경로
output_path = 'landmarks/guide_landmarks.pkl'  # 저장할 pickle 파일 경로
extract_and_save_hand_landmarks(video_path, output_path)
