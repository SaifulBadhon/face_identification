import cv2
import numpy as numpy
import os
face_classifier = cv2.CascadeClassifier('casced/data/haarcascade_frontalface_default.xml')
id = input("Enter The id: ")
os.mkdir("images/{}".format(id))


def face_extracter(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces ==() :
        return None
    
    for(x,y,w,h) in faces:
        cropped_faces = img[y:y+h,x:x+w]
    
    return cropped_faces



cap = cv2.VideoCapture(0)
count = 0

while True:
    ret,frame = cap.read()
    if face_extracter(frame) is not None:
        count+=1
        face = cv2.resize(face_extracter(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path = 'images/{}/'.format(id)+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count == 100:
        break
cap.release()
cv2.destroyAllWindows()
print("Data collection Done")



