import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import os
import itertools
import operator


def RunFaceDetectWithParam(modelpath = 'C:/Users/rithanya/Documents/Python/faces/AllEmp_ResizeEqBl.yml',
                  label_names = ['', 'akash', 'aravind', 'Harini', 'karthick_aravindan', 'Karthic_Chandran', 'keerthana', 'lalitha', 'Mani', 'Steve'],
                  trans = ['resize']):
    cascPath = "c:/Users/rithanya/Miniconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    log.basicConfig(filename='webcam.log',level=log.INFO)
    
    video_capture = cv2.VideoCapture(1)
    #anterior = 0
    
    #All Employee model
    lbp_model_path = modelpath

    
    # initialize a trained model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(lbp_model_path)
    
        
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
            if('resize' in trans):
                facegrab = cv2.resize(facegrab, (100, 100), interpolation = cv2.INTER_AREA)
            if('HE' in trans):
                facegrab = cv2.equalizeHist(facegrab)
            if('BL' in trans):
                facegrab = cv2.bilateralFilter(facegrab, 15, 80,80)
            pred = face_recognizer.predict(facegrab)
            text = label_names[pred[0]] + "," + str(pred[1])
            #label = face_recognizer.predict(facegrab)
            #text = str(label_names[label[0]]) # + ',' + str(label[1])
            #if(pred[1] < 60 or True):
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
        #if anterior != len(faces):
        #    anterior = len(faces)
        #    log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()) + " and label is" + str(label))
    
    
        # Display the resulting frame
        cv2.imshow('Video', frame)
    
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        # Display the resulting frame
        cv2.imshow('Video', frame)
    
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    


def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  #print ( L)
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]
