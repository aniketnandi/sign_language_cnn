"""
Real-time ASL hand-sign recognition from a webcam.

Pipeline (classical CV -> deep learning):
  1. Capture webcam frames and crop a fixed region of interest (ROI).
  2. Learn the empty background with MOG2 background subtraction.
  3. Clean the foreground mask with morphological opening/closing.
  4. Find the largest contour (the hand) and take its bounding box.
  5. Crop that box from the grayscale frame, resize to 28x28, normalize.
  6. Classify with the trained CNN (sign_language_cnn.h5).
  7. Smooth predictions over recent frames and display letter, confidence, FPS.

Usage:
  python realtime_demo.py
  Keep your hand OUT of the green box for the first ~2 seconds (background
  learning), then sign inside the box.

Keys:
  r  - re-learn the background (move your hand out first)
  q  - quit
"""

import time
from collections import Counter, deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- Configuration ----------------
MODEL_PATH = "sign_language_cnn.h5"
CAMERA_INDEX = 0
ROI = (350, 80, 300, 300)        # x, y, width, height of the signing box
BACKGROUND_FRAMES = 60           # frames used to learn the empty background
MIN_CONTOUR_AREA = 3000          # ignore small noise blobs
SMOOTHING_WINDOW = 10            # majority vote over the last N predictions
CONFIDENCE_THRESHOLD = 0.6       # below this, show "?" instead of a letter

# Sign Language MNIST labels: 0=A ... 24=Y. Label 9 (J) and Z need motion and
# are not in the dataset, so index 9 is never a valid prediction.
LETTERS = [chr(ord("A") + i) for i in range(25)]


def create_subtractor():
    """MOG2 background model; shadows disabled so the mask is binary."""
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50,
                                              detectShadows=False)


def hand_bounding_box(mask):
    """Return (x, y, w, h) of the largest foreground contour, or None."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, mask
    return cv2.boundingRect(largest), mask


def preprocess(gray_roi, box):
    """Crop the hand as a square, resize to 28x28, normalize like training."""
    x, y, w, h = box
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half = int(side * 0.65)
    rh, rw = gray_roi.shape
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, rw), min(cy + half, rh)
    crop = gray_roi[y0:y1, x0:x1]
    crop = cv2.equalizeHist(crop)
    crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
    return (crop.astype("float32") / 255.0).reshape(1, 28, 28, 1), crop


def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual mode on most Windows webcams
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 1.")

    subtractor = create_subtractor()
    frames_learned = 0
    recent = deque(maxlen=SMOOTHING_WINDOW)
    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)             # mirror view feels natural
        x, y, w, h = ROI
        roi = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        label_text = ""
        if frames_learned < BACKGROUND_FRAMES:
            # Phase 1: learn the empty background
            subtractor.apply(blurred, learningRate=-1)
            frames_learned += 1
            label_text = f"Learning background... {frames_learned}/{BACKGROUND_FRAMES}"
        else:
            # Phase 2: freeze the background model and detect the hand
            fg_mask = subtractor.apply(blurred, learningRate=0.0005)
            if cv2.countNonZero(fg_mask) > 0.6 * fg_mask.size:   # >60% "foreground" = lighting changed
                subtractor = create_subtractor()
                frames_learned = 0
                recent.clear()
                continue
            box, clean_mask = hand_bounding_box(fg_mask)
            cv2.imshow("Foreground mask", clean_mask)

            if box is not None:
                bx, by, bw, bh = box
                cv2.rectangle(roi, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                model_input, crop = preprocess(gray, box)
                cv2.imshow("Model input (28x28)",
                           cv2.resize(crop, (140, 140),
                                      interpolation=cv2.INTER_NEAREST))

                probs = model.predict(model_input, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf >= CONFIDENCE_THRESHOLD and idx != 9:
                    recent.append(idx)

                if recent:
                    voted = Counter(recent).most_common(1)[0][0]
                    label_text = f"{LETTERS[voted]}  ({conf:.0%})"
                else:
                    label_text = "?"
            else:
                recent.clear()
                label_text = "No hand detected"

        # FPS (exponential moving average for a stable readout)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("ASL Real-Time Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            subtractor = create_subtractor()
            frames_learned = 0
            recent.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
