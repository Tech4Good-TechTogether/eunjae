import cv2
import mediapipe as mp
import numpy as np
import pickle

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# KNN 모델 로드
with open('landmarks/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# 실시간 영상 예측
def predict_from_webcam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                # 예측
                prediction = knn.predict([hand_data])[0]

                # 피드백 제공
                if prediction == 0:
                    feedback = "올바른 자세입니다."
                else:
                    feedback = "엄지와 검지를 더 붙이세요."

                # 피드백을 화면에 표시
                cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 랜드마크를 프레임에 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실시간 예측 실행
predict_from_webcam()
