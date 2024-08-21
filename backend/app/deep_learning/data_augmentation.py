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

def augment_dataset(guide_landmarks, guide_labels, output_path, noise_level=8, rotation_range=(-10, 10)):
    augmented_landmarks = []
    augmented_labels = []

    for i, (landmarks_pair, label) in enumerate(zip(guide_landmarks, guide_labels)):
        left_hand = landmarks_pair[0]
        right_hand = landmarks_pair[1]

        if left_hand is not None:
            augmented_left = augment_landmarks([left_hand], noise_level, rotation_range)
            augmented_landmarks.extend([[left, right_hand] for left in augmented_left])
            augmented_labels.extend([label] * len(augmented_left))

        if right_hand is not None:
            augmented_right = augment_landmarks([right_hand], noise_level, rotation_range)
            augmented_landmarks.extend([[left_hand, right] for right in augmented_right])
            augmented_labels.extend([label] * len(augmented_right))

    # 증강된 데이터를 저장
    with open(output_path, 'wb') as f:
        pickle.dump({"landmarks": augmented_landmarks, "labels": augmented_labels}, f)

    print(f"Saved augmented landmarks and labels to {output_path}")

# 예제 실행
output_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks_with_labels_further_augmented.pkl')

augment_dataset(guide_landmarks, guide_labels, output_path)
