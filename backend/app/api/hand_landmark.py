import time
import cv2
import mediapipe as mp

def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detect hand landmarks
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only works on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # Draw landmarks on the hand
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # checking video frame rate
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
