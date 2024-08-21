import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# 가이드 영상의 올바른 좌표를 담은 pickle 파일 로드
guide_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks_with_labels_augmented.pkl')
with open(guide_landmarks_path, 'rb') as f:
    guide_data = pickle.load(f)
    guide_landmarks = guide_data["landmarks"]
    guide_labels = guide_data["labels"]

# 잘못된 동작 데이터를 담은 pickle 파일 로드
incorrect_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'incorrect_landmarks_with_labels.pkl')
with open(incorrect_landmarks_path, 'rb') as f:
    incorrect_data = pickle.load(f)
    incorrect_landmarks = incorrect_data['landmarks']
    incorrect_labels = incorrect_data['labels']

# 올바른 동작과 잘못된 동작 데이터를 결합
X = []
y = []

# 올바른 동작 데이터 추가
for frame_landmarks, label in zip(guide_landmarks, guide_labels):
    for hand_landmarks in frame_landmarks:
        if len(hand_landmarks) > 0:
            X.append(hand_landmarks.flatten())  # 21x3 -> 63차원 벡터
            y.append(label)

# 잘못된 동작 데이터 추가
for frame_landmarks, label in zip(incorrect_landmarks, incorrect_labels):
    for hand_landmarks in frame_landmarks:
        if len(hand_landmarks) > 0:
            X.append(hand_landmarks.flatten())  # 21x3 -> 63차원 벡터
            y.append(label)

X = np.array(X)
y = np.array(y)

# 데이터셋을 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 증강 및 균형
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 데이터 표준화 (평균을 0으로, 분산을 1로)
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# 더 복잡한 SVM 모델 및 하이퍼파라미터 설정
svc = SVC(C=10, gamma=0.01, kernel='rbf', random_state=42)  # C와 gamma 값을 높임
svc.fit(X_resampled, y_resampled)

# 테스트 데이터 예측
y_pred = svc.predict(X_test)

# 성능 평가
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 새로운 데이터에 대한 예측 함수
def predict_hand_pose(new_landmarks):
    new_landmarks_flatten = new_landmarks.flatten().reshape(1, -1)
    new_landmarks_scaled = scaler.transform(new_landmarks_flatten)
    prediction = svc.predict(new_landmarks_scaled)
    return prediction[0]  # 예측된 라벨 반환
