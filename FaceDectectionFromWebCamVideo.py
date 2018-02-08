import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import os



cascPath = "c:/Users/rithanya/Miniconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(1)
anterior = 0

#model specific information
#model_file_path = 'C:/Users/rithanya/Documents/Python/faces/DSEmp_02_08_19.yml'
#label_names = ['', 'akash', 'aravind', 'Harini', 'karthick_aravindan', 'Karthic_Chandran', 'keerthana', 'lalitha']


#All Employee model
model_file_path = 'C:/Users/rithanya/Documents/Python/faces/AllEmp_02_08_19.yml'
label_names = ['', 'akash', 'aravind', 'Harini', 'karthick_aravindan', 'Karthic_Chandran', 'keerthana', 'lalitha', 'Mani', 'Steve']

# initialize a trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_file_path)
    
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    

    # Classify faces and draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #continue
        facegrab = gray[y:y+w, x:x+h]
        label = face_recognizer.predict(facegrab)
        text = str(label_names[label[0]]) # + ',' + str(label[1])
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()) + " and label is" + str(label))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
