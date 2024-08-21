import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# 가이드 영상의 올바른 좌표를 담은 pickle 파일 로드
# guide_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks_with_labels_augmented.pkl')
# with open(guide_landmarks_path, 'rb') as f:
#     guide_data = pickle.load(f)
#     guide_landmarks = guide_data["landmarks"]
#     guide_labels = guide_data["labels"]

# 잘못된 동작 데이터를 담은 pickle 파일 로드
incorrect_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'incorrect_landmarks_with_labels.pkl')
with open(incorrect_landmarks_path, 'rb') as f:
    incorrect_data = pickle.load(f)
    incorrect_landmarks = incorrect_data['landmarks']
    incorrect_labels = incorrect_data['labels']

# 증강된 데이터를 담은 pickle 파일 로드
augmented_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'guide_landmarks_with_labels_further_augmented.pkl')
with open(augmented_landmarks_path, 'rb') as f:
    augmented_data = pickle.load(f)
    augmented_landmarks = augmented_data['landmarks']
    augmented_labels = augmented_data['labels']

# 올바른 동작, 잘못된 동작 및 증강된 동작 데이터를 결합
X = []
y = []

# 올바른 동작 데이터 추가
# for frame_landmarks, label in zip(guide_landmarks, guide_labels):
#     for hand_landmarks in frame_landmarks:
#         if len(hand_landmarks) > 0:
#             X.append(hand_landmarks.flatten())  # 21x3 -> 63차원 벡터
#             y.append(label)

# 잘못된 동작 데이터 추가
for frame_landmarks, label in zip(incorrect_landmarks, incorrect_labels):
    for hand_landmarks in frame_landmarks:
        if len(hand_landmarks) > 0:
            X.append(hand_landmarks.flatten())  # 21x3 -> 63차원 벡터
            y.append(label)

# # 증강된 동작 데이터 추가
for frame_landmarks, label in zip(augmented_landmarks, augmented_labels):
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

# 여러 모델을 비교할 리스트
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=6),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    # "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# 모델을 학습시키고 성능을 비교
best_model = None
best_score = 0
for name, model in models.items():
    # 교차 검증을 통해 모델의 성능 평가
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    print(f"{name} 평균 교차 검증 정확도: {mean_score:.4f}")
    
    # 가장 높은 성능을 가진 모델을 저장
    if mean_score > best_score:
        best_score = mean_score
        best_model = model

# 선택된 최적 모델로 학습 및 평가
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n선택된 모델:", best_model.__class__.__name__)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 새로운 데이터에 대한 예측 함수
def predict_hand_pose(new_landmarks):
    new_landmarks_flatten = new_landmarks.flatten().reshape(1, -1)
    new_landmarks_scaled = scaler.transform(new_landmarks_flatten)
    prediction = best_model.predict(new_landmarks_scaled)
    return prediction[0]  # 예측된 라벨 반환
