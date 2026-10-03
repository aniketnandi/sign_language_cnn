"""
Live ASL letter recognition using MediaPipe hand landmarks.

Why landmarks instead of pixels:
  The pixel CNN (trained on Sign MNIST) scores 98.7% on its test set but fails
  on live webcam video because of domain shift (different lighting, camera,
  background, and forearm in the crop). MediaPipe locates 21 hand joints
  regardless of background, and a classifier trained on those joints from
  YOUR webcam generalizes far better.

Three modes:
  python landmark_asl.py collect   # record labeled landmark samples
  python landmark_asl.py train     # train + evaluate a classifier
  python landmark_asl.py run       # live recognition with FPS

Setup (one time):
  pip install mediapipe scikit-learn joblib "opencv-python<5"
  Download the hand model next to this script:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Collect controls:
  press a letter key (a-z)  -> start recording samples for that letter
  SPACE                     -> stop recording
  ESC                       -> quit and save
"""

import csv
import os
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_TASK = "hand_landmarker.task"
DATA_CSV = "landmarks.csv"
CLASSIFIER = "landmark_model.joblib"
SMOOTHING_WINDOW = 8
CONFIDENCE_THRESHOLD = 0.6

# Hand skeleton connections for drawing (MediaPipe's 21-point layout)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def create_detector():
    if not os.path.exists(MODEL_TASK):
        sys.exit(f"Missing {MODEL_TASK}. Download it first (see the top of this file).")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(detector, frame, start_time):
    """Return 21 (x, y) pixel points for one hand, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.array([(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]])


def normalize(points):
    """Translate to the wrist and scale by hand size -> position/scale invariant."""
    rel = points - points[0]
    scale = np.max(np.linalg.norm(rel, axis=1)) or 1.0
    return (rel / scale).flatten()          # 42 features


def draw_hand(frame, points):
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)),
                 (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, tuple(p.astype(int)), 3, (0, 0, 255), -1)


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")
    return cap


def collect():
    detector, cap, start = create_detector(), open_camera(), time.time()
    new_file = not os.path.exists(DATA_CSV)
    f = open(DATA_CSV, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(42)])

    recording, counts = None, Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        points = detect(detector, frame, start)
        if points is not None:
            draw_hand(frame, points)
            if recording:
                writer.writerow([recording] + normalize(points).round(5).tolist())
                counts[recording] += 1

        status = f"REC {recording}: {counts[recording]}" if recording else "Press a letter to record"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (255, 255, 0), 2)
        cv2.imshow("Collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                         # ESC
            break
        if key == 32:                         # SPACE
            recording = None
        elif ord("a") <= key <= ord("z"):
            recording = chr(key).upper()

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Samples this session:", dict(counts))


def train():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    rows = list(csv.reader(open(DATA_CSV)))[1:]
    y = np.array([r[0] for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    print(f"{len(X)} samples, {len(set(y))} letters: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    print(f"Held-out accuracy: {clf.score(X_test, y_test):.3f}")
    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, CLASSIFIER)
    print(f"Saved {CLASSIFIER}")


def run():
    import joblib
    clf = joblib.load(CLASSIFIER)
    detector, cap, start = create_detector(), open_camera(), time.time()
    recent = deque(maxlen=SMOOTHING_WINDOW)
    prev, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        points = detect(detector, frame, start)
        text = "No hand"
        if points is not None:
            draw_hand(frame, points)
            probs = clf.predict_proba([normalize(points)])[0]
            idx = int(np.argmax(probs))
            if probs[idx] >= CONFIDENCE_THRESHOLD:
                recent.append(clf.classes_[idx])
            if recent:
                letter = Counter(recent).most_common(1)[0][0]
                text = f"{letter} ({probs[idx]:.0%})"
        else:
            recent.clear()

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.imshow("ASL Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27:       # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    modes = {"collect": collect, "train": train, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        sys.exit("Usage: python landmark_asl.py [collect|train|run]")
    modes[sys.argv[1]]()
