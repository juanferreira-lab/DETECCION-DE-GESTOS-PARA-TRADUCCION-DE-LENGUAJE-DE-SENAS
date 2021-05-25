# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:49:55 2021

@author: keybl
"""

import cv2
import numpy as np
import imutils
import numpy as np
import cv2
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import load_model

class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'NOTHING', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

bg = None

color_fingers = (0, 255, 255)
while True:
    ret, frame = cap.read()
    if ret == False: break
    # Redimensionar la imagen para que tenga un ancho de 640
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()
    
    load_img("prueba.jpg",target_size=(40,40))
    image=load_img("prueba.jpg",target_size=(40,40))
    image=img_to_array(image) 
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    prediction=model.predict(prediction_image)
    
    print(prediction)
    print(np.sum(prediction))
    print(np.argmax(prediction))
    print(class_names[np.argmax(prediction)])
    letrica=class_names[np.argmax(prediction)]
    
    if bg is not None:
        # Determinar la región de interés
        ROI = frame[50:250, 50:250]
        cv2.rectangle(frame, (50 - 2, 50 - 2), (270 + 2, 300 + 2), color_fingers, 1)
        cv2.putText(frame, letrica,(50 - 2, 50 - 2),cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),5)
       
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        # Región de interés del fondo de la imagen
        bgROI = bg[50:250, 50:250]
        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 7)

        cv2.imshow('th', th)
        cv2.imwrite('prueba.jpg', ROI)
        JJ = cv2.imread('prueba.jpg')
        
    cv2.imshow('Frame', frame)
    
    k = cv2.waitKey(20)
    
    ###prediccon 
    
    ##capa para mostrar letra 
    
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()