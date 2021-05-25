# DETECCIÓN DE GESTOS PARA TRADUCCIÓN DE LENGUAJE DE SEÑAS

Esta IA programada en lenguaje Python tiene  la capacidad de detectar los diferentes gestos que se pueden hacer con la mano  en el lenguaje de señas aparte de esto es capas de traducirlos y mostrar su significado en tiempo real usando el método de redes neuronales.

- Este repositorio contiene:
- Códigos fuente:
  - Entrenamiento
   - Testeo con cámara
-Link base de datos

# Requerimiento de Hardware:

-   Cámara
- 8 GB Ram
- 6 a 8 GB de espacio en el disco
- Entorno de desarrollo de lenguaje Python (Spyder)

# Requerimiento de Software:
Se requiere instalar las siguientes librerías en caso de no poseerlas en el sistema:
-   Python 3.8.5
    -   sklearn
    -   openCV
    -   Pandas
    -   numpy
    -   TensorFlow
    -    Imutils 
    -   Keras
    
# Instalar Librerías

- `'pip install numpy'`
- `'pip install -U scikit-learn'`
- `'pip install opencv-python'`
- `'pip install pandas'`
- `'pip install tensorflow'`
- `'pip install imutils'`
- `'pip install keras'`

# Uso Carpetas
Se requiere tener todo en una misma carpeta para que los códigos funcionen correctamente, para el entrenamiento se necesita crear una carpeta llamada "data"
en esta carpeta se guardaran las carpetas de la base de datos que contienen la imágenes de entrenamiento y testeo estas se guardaran como training_set y test_set.

# Configuración Código 
- para poder usar el código es necesario cumplir lo anterior mente mencionado sobre las carpetas cuando se tenga esto se abre el código de "entrenamiento_proyecto_IA" en el Spyder y se espera a que este termine.

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

- una vez terminado de correr el código este creara una carpeta "modelo" dentro del directorio donde este guardado y esta contendrá dos archivos .h5 llamados "modelo" y "pesos", con estos archivos se procede a abrir el código "camara_proyecto_IA".


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

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

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




- Este nos desplegara una ventana con la imagen de la cámara en esta ventana presionamos la letra "i" (tener en cuenta que no este puesto el botón de Mayúsculas) con esto aparecerá un recuadro amarillo donde se deben hacer los gestos una vez terminado se presiona la letra "q" para cerrar la venta de la cámara y finalizar el código.
