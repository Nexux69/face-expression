result = None
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model
#Variables
result = None
Data = None
frame = None
Model = load_model("EmojiScavenger.keras")
# Find haar cascade to draw bounding box around face
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")#Replace the Path
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary mapping class labels with corresponding emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
def getPredictionsFromModel():
  global result
  global frame
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = Model.predict(cropped_img)
    result = emotion_dict.get(int(np.argmax(prediction)))
    cv2.putText(frame, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame, (800, 600)))
def getResults():
  return result
cap = cv2.VideoCapture(0)
while True:
  ret, frame = cap.read()
  getPredictionsFromModel()
  result = getResults()
  print(' Facial Expression' + str(result))
  cv2.imshow("Live Camera", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
  		break
cap.release()
cv2.destroyAllWindows()