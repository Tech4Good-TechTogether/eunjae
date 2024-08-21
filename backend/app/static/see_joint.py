import pickle
import os
import numpy as np

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# 저장된 잘못된 랜드마크와 라벨이 있는 파일 경로
incorrect_landmarks_path = os.path.join(current_dir,  'landmarks', 'incorrect_landmarks_with_labels.pkl')

# 데이터를 로드하여 확인
with open(incorrect_landmarks_path, 'rb') as f:
    incorrect_landmarks_with_labels = pickle.load(f)

# 라벨 데이터의 분포 확인
# labels = incorrect_landmarks_with_labels['labels']
# label_distribution = {label: labels.count(label) for label in set(labels)}

print(np.array(incorrect_landmarks_with_labels['landmarks']).shape)
