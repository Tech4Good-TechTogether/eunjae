import pickle
import numpy as np
import os

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# static/landmarks 폴더의 절대 경로를 계산
landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks_with_labels_augmented.pkl')

# 가이드 영상의 올바른 좌표를 담은 pickle 파일을 로드
with open(landmarks_path, 'rb') as f:
    guide_data = pickle.load(f)
    guide_landmarks = guide_data["landmarks"]
    guide_labels = guide_data["labels"]

def generate_labeled_incorrect_data(guide_landmarks, noise_range=(8, 10)):
    incorrect_landmarks = []
    labels = []

    for frame in guide_landmarks:
        for hand_index in range(2):  # 각 프레임에서 두 손에 대해 반복
            hand_landmarks = frame[hand_index]
            if len(hand_landmarks) == 0:
                continue

            # 손이 오른손인지 왼손인지 판별
            is_right_hand = hand_landmarks[0][0] < hand_landmarks[9][0]

            # 랜덤 노이즈 생성 (4와 10 사이의 랜덤 값)
            noise_level = np.random.uniform(noise_range[0], noise_range[1])

            # 손에 대한 잘못된 동작 시뮬레이션
            if is_right_hand:
                # 1. 오른손 엄지와 검지가 충분히 맞닿지 않음
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][4][0] += noise_level  # 엄지를 오른쪽으로 이동
                incorrect_frame[hand_index][8][0] -= noise_level  # 검지를 왼쪽으로 이동
                incorrect_landmarks.append(incorrect_frame)
                labels.append("오른손 엄지와 검지가 충분히 맞닿지 않음")
                    
                # 2. 오른손 엄지 손가락이 과도하게 굽혀짐
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][4][2] += noise_level  # 엄지의 Z 좌표를 증가시켜 굽힘 효과
                incorrect_landmarks.append(incorrect_frame)
                labels.append("오른손 엄지 손가락이 과도하게 굽혀짐")
                
                # 3. 오른손 검지 손가락이 비정상적으로 치우침
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][8][0] += noise_level  # 검지를 오른쪽으로 이동
                incorrect_landmarks.append(incorrect_frame)
                labels.append("오른손 검지 손가락이 비정상적으로 치우침")
            else:
                # 1. 왼손 엄지와 검지가 충분히 맞닿지 않음
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][4][0] -= noise_level  # 엄지를 왼쪽으로 이동
                incorrect_frame[hand_index][8][0] += noise_level  # 검지를 오른쪽으로 이동
                incorrect_landmarks.append(incorrect_frame)
                labels.append("왼손 엄지와 검지가 충분히 맞닿지 않음")

                # 2. 왼손 엄지 손가락이 과도하게 굽혀짐
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][4][2] += noise_level  # 엄지의 Z 좌표를 증가시켜 굽힘 효과
                incorrect_landmarks.append(incorrect_frame)
                labels.append("왼손 엄지 손가락이 과도하게 굽혀짐")
                
                # 3. 왼손 검지 손가락이 비정상적으로 치우침
                incorrect_frame = np.copy(frame)
                incorrect_frame[hand_index][8][0] -= noise_level  # 검지를 왼쪽으로 이동
                incorrect_landmarks.append(incorrect_frame)
                labels.append("왼손 검지 손가락이 비정상적으로 치우침")

    return incorrect_landmarks, labels

# 예제 실행: 잘못된 동작 좌표 생성
incorrect_landmarks, labels = generate_labeled_incorrect_data(guide_landmarks)

incorrect_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'incorrect_landmarks_with_labels.pkl')

# 잘못된 좌표와 라벨을 저장
with open(incorrect_landmarks_path, 'wb') as f:
    pickle.dump({"landmarks": incorrect_landmarks, "labels": labels}, f)

print("Generated labeled incorrect landmarks and saved to file.")
