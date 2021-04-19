from cv2 import cv2
import numpy as np 
import pickle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smiles_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels = {"person_name":1}
with open("labels.pickle", 'rb') as f:
  og_labels = pickle.load(f)
  labels = {v:k for k, v in og_labels.items()}
recognizer.read("trainner.yml")
cap = cv2.VideoCapture(0)
while True:
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
  for (x,y,w,h) in faces:
    #print(x,y,w,h)
    roi_gray = gray[x:x+w, y:y+h]
    roi_color = frame[x:x+w, y:y+h]
    id_, conf = recognizer.predict(roi_gray)
    if conf>=4 : #\ and conf<=85 :
      #print(id_)
      #print(labels[id_])
      font = cv2.FONT_HERSHEY_SIMPLEX
      name = labels[id_]
      color = (255,255,255)
      stroke = 2
      cv2.putText(frame,name,(x,y),font, 1,color,stroke,cv2.LINE_AA)

    img_item = "my-img.png"
    cv2.imwrite(img_item,roi_gray)
    color = (255,0,0)
    stroke = 2
    cv2.rectangle(frame,(x,y),(x+w, y+h), color, stroke)
    smiles = smiles_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in smiles :
      cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
  cv2.imshow("frame",frame)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()