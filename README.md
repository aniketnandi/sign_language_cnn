# Sign Language Recognition with a CNN

A convolutional neural network built in TensorFlow/Keras that classifies static American Sign Language (ASL) hand signs from grayscale images. It reaches **98.72% accuracy** on a held-out test set of 7,172 images.

![Sample images](images/sample_images.png)

## Results

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 99.95% | **98.72%** |
| Loss | 0.0129 | 0.0478 |
| Macro F1 | – | 0.99 |

![Confusion matrix](images/confusion_matrix.png)

### Where the model struggles

Most classes score a perfect 1.00 F1. The errors are concentrated in a few pairs of hand shapes that look similar at 28×28 resolution:

| True letter | Recall | Most often confused with |
|---|---|---|
| T | 0.85 | G, X |
| N | 0.93 | M |
| Y | 0.94 | I |
| K | 0.96 | U |

N and M, for example, differ only in how many fingers fold over the thumb, which is hard to resolve in such small images.

## Dataset

[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (Kaggle):

- **27,455** training images and **7,172** test images
- 28×28 grayscale, stored as flattened pixel rows in CSV files
- **24 classes**, covering A–Y except J. J and Z are left out because they require motion.

The classes are roughly balanced, at about 950–1,300 training images each.

## Approach

**Preprocessing**
- Pixel values normalized to [0, 1] and reshaped to (28, 28, 1)
- Labels one-hot encoded
- Training data split 80/20 into train and validation sets (`random_state=42`)

**Data augmentation** (`ImageDataGenerator`)
- Rotation ±10°, zoom 10%, horizontal and vertical shifts of 10%

**Model architecture**

```
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
Flatten → Dense(128, ReLU) → Dense(softmax)
```

**Training**
- Adam optimizer with categorical cross-entropy loss
- Batch size 64, 10 epochs

Validation accuracy rose from 83.7% after the first epoch to 99.95% by the tenth, with validation loss falling steadily the whole time.

![Training curves](images/training_curves.png)

## Getting started

1. Download `sign_mnist_train.csv` and `sign_mnist_test.csv` from the Kaggle link above and place them in a `Sign Language MNIST/` folder.
2. Install the dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
   ```
3. Open the notebook, update the two CSV paths in the loading cell to point at your copy, and run all cells.

The trained model is saved as `sign_language_cnn.h5`.

## Real-Time Webcam Demo

`realtime_demo.py` runs the trained CNN on live webcam video at ~17 FPS on CPU.
It uses classical CV to find the hand first: MOG2 background subtraction,
morphological filtering, and contour detection, then crops the hand to 28×28
and classifies it.

```bash
   pip install -r requirements.txt
   python realtime_demo.py
```
Keep your hand out of the green box for ~2 seconds while it learns the
background, then sign inside it. Press `r` to relearn the background, `q` to quit.

**Note:** live accuracy is lower than the 98.7% test accuracy because webcam
frames differ from Sign MNIST images (domain shift).

## Live Recognition with Hand Landmarks

### Why a second approach?
The pixel-based CNN scores 98.7% on the Sign MNIST test set, but almost 0% on live webcam video. Debugging showed three causes:
- **Domain shift:** webcam lighting, camera, and background differ from Sign MNIST images
- **Forearm in the crop:** Sign MNIST shows only the hand; live crops included the arm
- **Confident misclassification:** the model was often 99–100% sure and wrong, and biased toward "O"

### The fix
`landmark_asl.py` replaces pixels with hand geometry:
1. **MediaPipe Hands** finds 21 hand-joint landmarks in each frame, regardless of background
2. Landmarks are normalized to the wrist and scaled by hand size, which makes them position- and scale-invariant (42 features)
3. A **Random Forest** trained on 5,600+ self-collected samples across all 26 letters classifies the handshape
4. Predictions are smoothed with a majority vote over recent frames

### Results
- Recognized **all 26 letters** in live webcam testing at **~30 FPS** on CPU
- Similar handshapes (M/N/S/T, U/V/R) are less stable
- J and Z involve motion in ASL; a static-landmark model can only approximate them
- Held-out accuracy on recorded samples is 100%, but this is inflated because consecutive frames are nearly identical. Live testing is the meaningful metric.

### Run it
Download the MediaPipe hand model into this folder:
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

```bash
pip install -r requirements.txt
python landmark_asl.py collect   # record samples: press a letter key to record, SPACE to stop, ESC to save
python landmark_asl.py train     # train and evaluate the classifier
python landmark_asl.py run       # live recognition
```

## Possible improvements
- Sequence model (LSTM or temporal CNN) over landmark sequences to properly handle J and Z
- More training data from multiple people and lighting conditions to improve robustness
- PyTorch port and C++ deployment via ONNX and OpenCV DNN (in progress)
