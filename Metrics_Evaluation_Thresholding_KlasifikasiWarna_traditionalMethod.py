import json
import cv2
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth

def evaluate_predictions(ground_truth, predictions):
    y_true = [ground_truth[file] for file in ground_truth.keys()]
    y_pred = [predictions.get(file, 0) for file in ground_truth.keys()]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, accuracy

def detect_hand(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return 0
    
    lower_skin = np.array([0, 40, 50], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

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
            return 1

    return 0

def generate_predictions(dataset_path):
    predictions = {}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            predictions[file] = detect_hand(file_path)
    return predictions

if __name__ == "__main__":
    dataset_path = "datasetHand"
    ground_truth_path = "ground_truth.json"

    ground_truth = load_ground_truth(ground_truth_path)

    predictions = generate_predictions(dataset_path)

    precision, recall, accuracy = evaluate_predictions(ground_truth, predictions)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
