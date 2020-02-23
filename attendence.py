import numpy as np 
import cv2
import pickle

face_casced = cv2.CascadeClassifier('casced/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name" : 1}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_casced.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        id_,conf = recognizer.predict(roi_gray)
        print(conf)
        if conf>=45 and conf<=85:
            print(labels[id_])
        

        

        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()

