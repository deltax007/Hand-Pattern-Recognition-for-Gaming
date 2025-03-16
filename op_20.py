import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load hand detector
class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = int(maxHands)
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return self.results

    def findPosition(self, img):
        lmList = []
        bbox = None

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                x_min, y_min = min(x_min, cx), min(y_min, cy)
                x_max, y_max = max(x_max, cx), max(y_max, cy)

            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return lmList, bbox

# Load TensorFlow Model
model_path = r"C:\Users\omen\OneDrive\Desktop\op_p\test13\model\sssss\model.savedmodel"
model = tf.keras.models.load_model(model_path)

# Load labels
labels_path = r"C:\Users\omen\OneDrive\Desktop\op_p\test13\model\labels.txt"
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

detector = handDetector()

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect hands
    results = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    prediction_label = "None"
    confidence = 0.0

    if bbox:
        x, y, w, h = bbox
        offset = 20
        imgSize = 400

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgInput = cv2.resize(imgCrop, (300, 300))
            imgInput = (imgInput.astype("float32") / 127.5) - 1
            imgInput = np.expand_dims(imgInput, axis=0)

            predictions = model.predict(imgInput)
            index = np.argmax(predictions)
            prediction_label = labels[index]
            confidence = float(predictions[0][index])

    return jsonify({"prediction": prediction_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
