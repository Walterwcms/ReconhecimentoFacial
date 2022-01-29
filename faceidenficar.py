

import cv2
import pickle





video_capture = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#-----------------------------------------------------------------------------------------------------------------------
labels = {}
f = open("labels.pickle","rb")
og_labels  = pickle.load(f)
f.close()
    #reverter dicionario key e value
labels = {v:k for k,v in og_labels.items()}


#---------------padroes de reconhecimento----------------------------------------------------------
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#--------------------------------------------------------------------------------------------------

#contTempoAbrirPorta = 10

verde  = (0,255,0)
vermelho = (0,0,255)
corRetangulo = vermelho
nomeLabel = ""
corLetra = (0, 0, 255)
branco = (255,255,255)

while True:
    ret, frame = video_capture.read()

    #---------------tonalidade para facil identificacao-----------------
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    #----------identificar rosto--------------------------------------
    faces = faceCascade.detectMultiScale(
        cinza,
        scaleFactor= 1.1,
        minNeighbors=5,
        minSize=(30,30)
    )
    #------------------------------------------------------------------

    if(2 > len(faces) > 0):
        #---------------------mostrar o retangulo verde sobre o rosto------
        # x,y,w,h -->posicoes das faces capturadas
        for(x, y, w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), corRetangulo, 2)
            roi_gray = cinza[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            fonte = cv2.FONT_HERSHEY_SIMPLEX
            stroke = 2
            cv2.putText(frame, nomeLabel, (x, y), fonte, 1, corLetra, stroke, cv2.LINE_AA)

            if(conf <= 45 and (labels[id_] == "walter" or labels[id_] == "kleiton")):
                print('\033[92m'+"identificado - Porta aberta"+'\033[0m')
                corRetangulo = verde
                nomeLabel  = labels[id_]
                corLetra = branco
            else:
                corRetangulo = vermelho
                nomeLabel  = "desconhecido"
                corLetra = vermelho
            print(conf)

        #------------------------------------------------------------------
    if(1 < len(faces)):
        print("apenas uma pessoa")


#-----------------------------------------------------------------------------------------------------------------------



    cv2.imshow("video",frame)

    #---sair--------------------------------
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

video_capture.release()
cv2.destroyAllWindows()