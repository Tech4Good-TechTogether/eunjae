import cv2
import mediapipe as mp
import numpy as np
import pickle

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def augment_landmarks(landmarks, noise_level=8, rotation_range=(-10, 10)):
    augmented_landmarks = []

    for hand_landmarks in landmarks:
        # 원본 랜드마크 추가
        augmented_landmarks.append(hand_landmarks)

        # 1. 노이즈 추가
        noise = np.random.normal(0, noise_level / 1000, hand_landmarks.shape)  # 노이즈를 적절히 줄임
        noisy_landmarks = hand_landmarks + noise
        augmented_landmarks.append(noisy_landmarks)

        # 2. 좌우 반전
        mirrored_landmarks = np.copy(hand_landmarks)
        mirrored_landmarks[:, 0] = 1.0 - mirrored_landmarks[:, 0]  # X 좌표 반전
        augmented_landmarks.append(mirrored_landmarks)

        # 3. 회전 추가
        theta = np.radians(np.random.uniform(rotation_range[0], rotation_range[1]))  # -10도에서 10도 사이의 랜덤 회전
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_landmarks = np.copy(hand_landmarks)
        rotated_landmarks[:, :2] = np.dot(rotated_landmarks[:, :2], rotation_matrix)  # X, Y에 대해서만 회전 적용
        augmented_landmarks.append(rotated_landmarks)

    return augmented_landmarks

def extract_and_save_hand_landmarks_with_labels(video_path, output_path, label="올바른 동작"):
    # 영상 읽기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    all_landmarks = []
    all_labels = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 프레임을 BGR에서 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 랜드마크 탐지
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            frame_landmarks = {'left_hand': None, 'right_hand': None}  # 왼손과 오른손을 구분하여 저장

            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append([lm.x, lm.y, lm.z])
                hand_data = np.array(hand_data)

                # 손이 오른손인지 왼손인지 판별
                is_right_hand = hand_data[0][0] < hand_data[9][0]

                if is_right_hand:
                    frame_landmarks['right_hand'] = hand_data
                else:
                    frame_landmarks['left_hand'] = hand_data
            
            # 왼손과 오른손을 각각 augmentation한 후 저장
            augmented_landmarks = [None, None]  # 왼손과 오른손을 담기 위한 리스트
            if frame_landmarks['left_hand'] is not None:
                augmented_landmarks[0] = augment_landmarks([frame_landmarks['left_hand']])[0]
            if frame_landmarks['right_hand'] is not None:
                augmented_landmarks[1] = augment_landmarks([frame_landmarks['right_hand']])[0]
            
            # 왼손과 오른손의 정보를 모두 포함하여 한 프레임의 정보로 추가
            if augmented_landmarks:
                all_landmarks.append(augmented_landmarks)
                all_labels.append(label)
            
            # 랜드마크를 프레임에 그리기
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 랜드마크가 그려진 프레임을 화면에 표시
        cv2.imshow('Hand Landmarks', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # 추출한 랜드마크와 라벨을 pickle 파일로 저장
    if all_landmarks:  # 데이터가 있을 때만 저장
        with open(output_path, 'wb') as f:
            pickle.dump({"landmarks": all_landmarks, "labels": all_labels}, f)
        print(f"Saved augmented landmarks and labels to {output_path}")
    else:
        print("No landmarks were detected or saved.")
        
        
# 예제 실행
video_path = 'videos/hand_guide.mp4'  # 입력할 가이드 영상 파일 경로
output_path = 'landmarks/guide_landmarks_with_labels_augmented.pkl'  # 저장할 pickle 파일 경로
extract_and_save_hand_landmarks_with_labels(video_path, output_path)
