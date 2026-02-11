# file: main.py
# Raspberry Pi main: Camera (OpenCV Haar) + TFLite gender -> PCA9685 hand gestures (I2C)
#
# Prereqs (inside your venv):
#   pip install opencv-python numpy
#   pip install tflite-runtime          # on Pi (recommended)  OR  pip install tensorflow
#   pip install adafruit-blinka adafruit-circuitpython-pca9685
#
# Enable I2C:
#   sudo raspi-config  -> Interface Options -> I2C -> Enable
#
# Models expected:
#   ./models/haarcascade_frontalface_default.xml
#   ./models/GenderClass_06_03-20-08.tflite
#
# Run:
#   python main.py

import os
import time
from pathlib import Path

import cv2
import numpy as np

# --- TFLite interpreter (Pi: tflite_runtime, dev PC: tensorflow.lite) ---
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

# --- Hand control (PCA9685) ---
# Put the hand script in the same folder (or adjust import)
from hand_gestures_pca9685 import on_gender  # uses startup animation automatically

# ===== Paths =====
BASE = Path(__file__).resolve().parent
FACE_MODEL = BASE / "models" / "haarcascade_frontalface_default.xml"
GENDER_MODEL = BASE / "models" / "GenderClass_06_03-20-08.tflite"

# ===== Settings =====
CAM_INDEX = 0
FACE_MIN_SIZE = (60, 60)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

# Debounce to avoid spamming servos
SEND_MIN_INTERVAL_SEC = 0.6

# If no face seen for this long -> neutral
NO_FACE_TIMEOUT_SEC = 1.5


def load_face_detector():
    face_cascade = cv2.CascadeClassifier(str(FACE_MODEL))
    if face_cascade.empty():
        raise FileNotFoundError(f"Failed to load Haar cascade: {FACE_MODEL}")
    return face_cascade


def load_gender_model():
    interpreter = Interpreter(model_path=str(GENDER_MODEL))
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    return interpreter, in_details, out_details


def preprocess_face_for_model(face_bgr: np.ndarray) -> np.ndarray:
    # Model expects (1, 224, 224, 3) float32 in [0,1]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224))
    face_rgb = face_rgb.astype(np.float32) / 255.0
    return np.expand_dims(face_rgb, axis=0)


def predict_gender(interpreter, in_details, out_details, face_bgr: np.ndarray) -> str:
    x = preprocess_face_for_model(face_bgr)
    interpreter.set_tensor(in_details[0]["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_details[0]["index"])  # shape (1, 2) typically
    idx = int(np.argmax(out[0]))
    # Keep same convention you already used: idx==1 -> male else female
    return "male" if idx == 1 else "female"


def pick_largest_face(faces):
    # faces: list of (x,y,w,h)
    return max(faces, key=lambda r: r[2] * r[3])


def main():
    print("Loading models...")
    face_cascade = load_face_detector()
    interpreter, in_details, out_details = load_gender_model()

    print("Opening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    last_sent_label = None
    last_sent_time = 0.0
    last_face_time = time.time()

    print("Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
        )

        label = None

        if len(faces) > 0:
            (x, y, w, h) = pick_largest_face(faces)
            last_face_time = time.time()

            face = frame[y : y + h, x : x + w]
            label = predict_gender(interpreter, in_details, out_details, face)

            # UI overlay
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            # If no face for a while -> neutral
            if (time.time() - last_face_time) > NO_FACE_TIMEOUT_SEC:
                label = "neutral"

        # Send to hand with debounce
        if label is not None:
            now = time.time()
            if (label != last_sent_label) or ((now - last_sent_time) >= SEND_MIN_INTERVAL_SEC):
                on_gender(label)
                last_sent_label = label
                last_sent_time = now

        cv2.imshow("HRI: Face + Gender -> Hand", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()