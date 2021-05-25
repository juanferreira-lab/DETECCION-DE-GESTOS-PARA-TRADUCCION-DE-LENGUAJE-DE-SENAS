# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:12:25 2021

@author: keybl
"""

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

Directorio_Entrenamiento= './data/training_set'
Directorio_Testeo= './data/test_set'

Nombre_clase=[]
for file in os.listdir(Directorio_Entrenamiento):
    Nombre_clase+=[file]
print(Nombre_clase)
print(len(Nombre_clase))

N=[]
for i in range(len(Nombre_clase)):
    N+=[i]
    
mapeado=dict(zip(Nombre_clase,N)) 
mapeado_inverso=dict(zip(N,Nombre_clase)) 

def mapper(value):
    return mapeado_inverso[value]
dataset=[]
contador=0
for file in os.listdir(Directorio_Entrenamiento):
    path=os.path.join(Directorio_Entrenamiento,file)
    for im in os.listdir(path):
        imagen=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(40,40))
        imagen=img_to_array(imagen)
        imagen=imagen/255.0
        dataset+=[[imagen,contador]]
    contador=contador+1
    
testset=[]
contador=0
for file in os.listdir(Directorio_Testeo):
    path=os.path.join(Directorio_Testeo,file)
    for im in os.listdir(path):
        imagen=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(40,40))
        imagen=img_to_array(imagen)
        imagen=imagen/255.0
        testset+=[[imagen,contador]]
    contador=contador+1
    
data,labels0=zip(*dataset)
test,testlabels0=zip(*testset)

labels1=to_categorical(labels0)
labels=np.array(labels1)
data=np.array(data)
test=np.array(test)
trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.2,random_state=44)
print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)

Generador = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")
Modelo_preentrenado= tf.keras.applications.DenseNet201(input_shape=(40,40,3),include_top=False,weights='imagenet',pooling='avg')
Modelo_preentrenado.trainable = False

Entradas= Modelo_preentrenado.input
x3 = tf.keras.layers.Dense(128, activation='relu')(Modelo_preentrenado.output)
outputs3 = tf.keras.layers.Dense(28, activation='softmax')(x3)
modelo = tf.keras.Model(inputs=Entradas, outputs=outputs3)
modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=modelo.fit(Generador.flow(trainx,trainy,batch_size=32),validation_data=(testx,testy),epochs=15)


prediccion_testeo=modelo.predict(testx)
predict_testeo=np.argmax(prediccion_testeo,axis=1)
ground = np.argmax(testy,axis=1)
print(classification_report(ground,predict_testeo))

 
precision = his.history['accuracy']
valor_precision = his.history['val_accuracy']
perdidas= his.history['loss']
validacion_perdidas= his.history['val_loss']

epocas= range(len(precision))
plt.plot(epocas, precision, 'r', label='Accuracy of Training data')
plt.plot(epocas, valor_precision, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

epocas = range(len(perdidas))
plt.plot(epocas, perdidas, 'r', label='Loss of Training data')
plt.plot(epocas, validacion_perdidas, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

target_directorio = './modelo/'
if not os.path.exists(target_directorio):
  os.mkdir(target_directorio)
modelo.save('./modelo/modelo.h5')
modelo.save_weights('./modelo/pesos.h5')   




