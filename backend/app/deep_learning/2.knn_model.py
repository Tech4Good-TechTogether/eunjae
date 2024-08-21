import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# 원본 데이터 경로
incorrect_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'incorrect_landmarks_with_labels.pkl')

# 증강된 데이터 경로
augmented_landmarks_path = os.path.join(current_dir, '..', 'static', 'landmarks', 'augmented_landmarks_with_labels.pkl')

# 원본 데이터 로드
with open(incorrect_landmarks_path, 'rb') as f:
    incorrect_data = pickle.load(f)

landmarks = incorrect_data['landmarks']
labels = incorrect_data['labels']

# 증강된 데이터 로드
with open(augmented_landmarks_path, 'rb') as f:
    augmented_data = pickle.load(f)

augmented_landmarks = augmented_data['landmarks']
augmented_labels = augmented_data['labels']

# 원본 데이터와 증강된 데이터를 결합
landmarks.extend(augmented_landmarks)
labels.extend(augmented_labels)

# 2. 데이터 전처리
# landmarks는 (frame, hand_index, 21, 3) 형태로 되어 있으므로 이를 2D로 평탄화
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 표준화 (평균을 0으로, 분산을 1로)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. KNN 모델 학습
# k = 11
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print("k : ",k)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
for i in range(1,15):
    k = i  # KNN에서 고려할 이웃의 수
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("k : ",k)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 4. 예측 및 성능 평가


# 5. 새로운 데이터에 대한 예측 함수
def predict_hand_pose(new_landmarks):
    # 입력된 새로운 손동작의 랜드마크를 전처리하여 예측
    new_landmarks_flatten = new_landmarks.flatten().reshape(1, -1)  # 21x3 -> 63차원 벡터로 변환
    new_landmarks_scaled = scaler.transform(new_landmarks_flatten)  # 스케일링 적용
    prediction = knn.predict(new_landmarks_scaled)
    return prediction[0]  # 예측된 라벨 반환

# 예제 사용: 새로운 손동작 예측
# 새로운 랜드마크 데이터를 사용하여 동작이 올바른지 잘못되었는지 예측
# new_landmarks_example = ... (21x3 형태의 numpy 배열로 입력)
# prediction = predict_hand_pose(new_landmarks_example)
# print("Predicted Label:", prediction)
