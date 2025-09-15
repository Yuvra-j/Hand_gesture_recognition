import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

def predict_real_time():
    try:
        model = load_model("C:/Users/Yuvraj/gesture_model.h5")
        print("Model loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Please check the file path or save the model first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, c = frame.shape
                x_min = int(min([lm.x * w for lm in hand_landmarks.landmark]))
                y_min = int(min([lm.y * h for lm in hand_landmarks.landmark]))
                x_max = int(max([lm.x * w for lm in hand_landmarks.landmark]))
                y_max = int(max([lm.y * h for lm in hand_landmarks.landmark]))
                roi = frame[y_min:y_max, x_min:x_max]

                if roi.size > 0:
                    resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA) / 255.0
                    input_img = np.expand_dims(resized, axis=0)
                    pred = model.predict(input_img)
                    gesture_idx = np.argmax(pred)
                    gesture = gestures[gesture_idx]
                    confidence = np.max(pred) * 100
                    cv2.putText(frame, f'{gesture} ({confidence:.1f}%)', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_real_time()