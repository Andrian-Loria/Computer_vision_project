import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Function to set volume on Windows
def set_volume_windows(volume_percent):
    volume_percent = max(0, min(100, int(volume_percent)))
    
    # Get the audio endpoint (default system audio device)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, 1, None)
    volume_control = interface.QueryInterface(IAudioEndpointVolume)

    # Set volume level
    volume_scalar = volume_percent / 100.0
    volume_control.SetMasterVolumeLevelScalar(volume_scalar, None)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

app = Flask(__name__)

current_volume = 50

def gen():
    global current_volume
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                index_tip = hand_landmarks.landmark[8]  # Index tip
                palm_base = hand_landmarks.landmark[0]  # Palm base
                middle_tip = hand_landmarks.landmark[12]  # Middle tip

                h, w, _ = frame.shape
                thumb_pos = np.array([int(thumb_tip.x * w), int(thumb_tip.y * h)])
                index_pos = np.array([int(index_tip.x * w), int(index_tip.y * h)])
                palm_pos = np.array([int(palm_base.x * w), int(palm_base.y * h)])
                middle_pos = np.array([int(middle_tip.x * w), int(middle_tip.y * h)])

                cv2.circle(frame, tuple(thumb_pos), 10, (255, 0, 0), -1)
                cv2.circle(frame, tuple(index_pos), 10, (255, 0, 0), -1)
                cv2.circle(frame, tuple(palm_pos), 10, (0, 255, 0), -1)
                cv2.circle(frame, tuple(middle_pos), 10, (0, 255, 0), -1)

                thumb_index_gap = np.linalg.norm(thumb_pos - index_pos)
                hand_length = np.linalg.norm(palm_pos - middle_pos)

                normalized_gap = thumb_index_gap / hand_length if hand_length != 0 else 0
                current_volume = np.interp(normalized_gap, [0.1, 0.5], [0, 100])

                set_volume_windows(current_volume)

                cv2.putText(frame, f'Volume: {int(current_volume)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', frame_rgb)
        if not ret:
            print("Error: Failed to encode frame")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_volume')
def get_volume():
    return jsonify({"volume": int(current_volume)})

if __name__ == '__main__':
    app.run(debug=True)
