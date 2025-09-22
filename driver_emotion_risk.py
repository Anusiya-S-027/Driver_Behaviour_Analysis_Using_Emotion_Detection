import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model

model_path = r"E:\Coding\Driver Behaviour Analysis\emotion_model.h5"
if not os.path.exists(model_path):
    url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    urllib.request.urlretrieve(url, model_path)

model = load_model(model_path, compile=False)

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
EMOTION_WEIGHTS = {"angry":0.9,"disgust":0.8,"fear":0.9,"happy":0.15,"sad":0.6,"surprise":0.5,"neutral":0.1}

def compute_risk(preds):
    return float(sum(p * EMOTION_WEIGHTS[e] for e,p in zip(EMOTIONS,preds)))

def risk_label(score):
    if score < 0.2: return "LOW"
    elif score < 0.45: return "MODERATE"
    elif score < 0.7: return "HIGH"
    else: return "CRITICAL"

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h,x:x+w]
        face = cv2.resize(face,(64,64))
        face = face.astype("float32")/255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face, verbose=0)[0]
        emotion_idx = int(np.argmax(preds))
        emotion = EMOTIONS[emotion_idx]
        prob = preds[emotion_idx]
        risk_score = compute_risk(preds)
        risk = risk_label(risk_score)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{emotion} ({prob:.2f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Risk: {risk}",(x+w+5,y+h//2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imshow("Driver Emotion Risk Monitor",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
