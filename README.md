# Driver Emotion Risk Monitoring

This project uses **TensorFlow** and **OpenCV** to detect a driver’s facial emotion in real-time and calculate a **risk factor** based on the detected emotion. The system captures live video from a webcam, predicts emotion for each detected face, and displays both the emotion and associated risk on the video feed.

---

## Features

* **Live webcam feed** with face detection.
* **Emotion recognition** using a pretrained Mini-XCEPTION model trained on FER2013 dataset.
* **Risk calculation** based on emotion intensity:

  * Angry, Disgust, Fear → high risk
  * Happy → low risk
  * Neutral, Sad, Surprise → moderate risk
* **Overlay display**:

  * Emotion and probability on top of the face box.
  * Risk factor aligned to the right of the face box.
* Automatic download of the pretrained model if not present.

---

## Requirements

* Python 3.10+ (or compatible)
* Packages:

```bash
pip install opencv-python numpy tensorflow
```

---

## Usage

1. Clone or copy the script `driver_emotion_risk.py` to your project folder.
2. Run the script:

```bash
python driver_emotion_risk.py
```

3. The first time, the pretrained model `emotion_model.h5` will be automatically downloaded.
4. Press **`q`** to quit the application.

---

## How It Works

1. Captures video frames from your webcam.
2. Detects faces using OpenCV Haar Cascades.
3. Preprocesses each face to `64x64` grayscale images.
4. Uses TensorFlow model to predict emotion probabilities.
5. Computes a **risk score** by weighting emotions.
6. Displays the **emotion label on top** of the face and **risk label to the side** of the bounding box.

---

## Emotion Risk Weighting

| Emotion  | Weight |
| -------- | ------ |
| angry    | 0.9    |
| disgust  | 0.8    |
| fear     | 0.9    |
| happy    | 0.15   |
| sad      | 0.6    |
| surprise | 0.5    |
| neutral  | 0.1    |

* Risk is categorized into: LOW, MODERATE, HIGH, CRITICAL based on score.

---

## Notes

* The model used is **Mini-XCEPTION** trained on FER2013.
* The application works best with a **well-lit environment**.
* Multiple faces in the frame are supported, each with its own emotion and risk label.
