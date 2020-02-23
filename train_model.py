import numpy as np
from PIL import Image
import os
import pickle
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_casced = cv2.CascadeClassifier('casced/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root)
            print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            
            id_ = label_ids[label]
            pil_image = Image.open(path).convert('L')
            image_array = np.asarray(pil_image,dtype=np.uint8)

            faces = face_casced.detectMultiScale(image_array ,scaleFactor=1.5,minNeighbors=5)

            
            r = image_array
            
            x_train.append(r)
            y_labels.append(id_)
                 
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)


recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")

