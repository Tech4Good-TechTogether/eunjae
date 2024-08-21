import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)
incorrect_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'incorrect_landmarks_with_labels.pkl')

# 잘못된 동작 데이터 로드
with open(incorrect_landmarks_path, 'rb') as f:
    incorrect_data = pickle.load(f)

landmarks = incorrect_data['landmarks']
labels = incorrect_data['labels']

# 데이터 전처리
X = []
y = []

for frame_landmarks, label in zip(landmarks, labels):
    for hand_landmarks in frame_landmarks:
        if len(hand_landmarks) > 0:
            X.append(hand_landmarks.flatten())  # 21x3 -> 63차원 벡터
            y.append(label)

X = np.array(X)
y = np.array(y)

# 데이터셋을 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화 (평균을 0으로, 분산을 1로)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 모델 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 새로운 데이터에 대한 예측 함수
def predict_hand_pose(new_landmarks):
    new_landmarks_flatten = new_landmarks.flatten().reshape(1, -1)
    new_landmarks_scaled = scaler.transform(new_landmarks_flatten)
    prediction = rf.predict(new_landmarks_scaled)
    return prediction[0]  # 예측된 라벨 반환
