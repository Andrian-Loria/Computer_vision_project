from flask import Flask, render_template, Response
import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0)
lower_skin = np.array([0, 40, 50], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
kernel = np.ones((5, 5), np.uint8)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
volume = interface.QueryInterface(IAudioEndpointVolume) 

last_stable_time = 0
stability_threshold = 3  
last_distance = None
confirmed = False

def get_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_stable_points(hull):
    stable_points = []
    for point in hull:
        if point[0][1] < 250:
            stable_points.append(tuple(point[0]))

    if len(stable_points) > 1:
        highest_point = max(stable_points, key=lambda x: x[1])
        lowest_point = min(stable_points, key=lambda x: x[1])
        return highest_point, lowest_point

    return None, None


def generate_frames():
    global last_stable_time, last_distance, confirmed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                hull = cv2.convexHull(cnt)
                highest_point, lowest_point = get_stable_points(hull)

                if highest_point and lowest_point:
                    cv2.circle(frame, highest_point, 5, (0, 0, 255), -1)
                    cv2.circle(frame, lowest_point, 5, (0, 255, 0), -1)
                    distance = get_distance(highest_point, lowest_point)

                    if last_distance is not None and abs(distance - last_distance) < 10:
                        if time.time() - last_stable_time >= stability_threshold:
                            confirmed = True
                    else:
                        last_stable_time = time.time()

                    last_distance = distance
                    volume_level = max(0.0, min(1.0, (distance / 200.0)))
                    volume.SetMasterVolumeLevelScalar(volume_level, None)
                    cv2.putText(frame, f"Volume: {int(volume_level * 100)}%", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if confirmed:
                        cv2.putText(frame, "Posisi Stabil Terkonfirmasi!", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cap.release()
                        break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

