import os
from PIL import Image
import numpy as np
import cv2
import pickle

#---------------padroes de reconhecimento----------------------------------------------------------
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()


#--------------------procurar imagens na base de dados------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
imagens_dir = os.path.join(base_dir, "imagens")


current_id = 0
label_ids = {}
y_labels = []
x_train = []

print(". . . Espera um pouco, estou treinando, tenha paciencia.")

for root, dirs , files in os.walk(imagens_dir):
    for file in files:
        if(file.endswith("png") or file.endswith("jpg")):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #y_labels.append(label)
            #x_train.append(path)

            #-----colocar imagens em array, formato de pixel--------------
            pil_image = Image.open(path).convert("L") #escala de cizento
            image_array = np.array(pil_image, "uint8")

            # ----------identificar rosto--------------------------------------------------
            faces = faceCascade.detectMultiScale(image_array,scaleFactor=1.1,minNeighbors=5)


            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#-----------------------------------------------------------------------------------------------------------------------

print(label_ids)
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

print("Terminou, gracas a deus :) ")


