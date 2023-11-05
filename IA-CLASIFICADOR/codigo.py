import tensorflow as tf
from keras import layers,models
import os
import numpy as np
import cv2
import random
#dimension 264x264

width=300
height=300
rut_entrada="C:/Users/pedro/Desktop/IA-CLASIFICADOR/IMAGENES/entrada/"
rut_prueba="C:/Users/pedro/Desktop/IA-CLASIFICADOR/IMAGENES/test/"
#Effusion/Infiltration/NoFinding/Pneumothorax
entrenamiento_x=[]
entrenamiento_y=[]

# for i in os.listdir(rut_entrada):
#     for j in os.listdir(rut_entrada+i):
#         img=cv2.imread(rut_entrada+i+"/"+j)
#         resized_image=cv2.resize(img,(width,height))
        
#         entrenamiento_x.append(resized_image)
#         if i=="Effusion":
#             entrenamiento_y.append([0,0,0,1])
#         elif i=="Infiltration":
#             entrenamiento_y.append([0,0,1,0])
#         elif i=="NoFinding":
#             entrenamiento_y.append([0,1,0,0])
#         elif i=="Pneumothorax":
#             entrenamiento_y.append([1,0,0,0])

# x_data=np.array(entrenamiento_x)
# y_data=np.array(entrenamiento_y)

# model=tf.keras.Sequential([
#     layers.Conv2D(32,3,3,input_shape=(width,height,3)),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(pool_size=(2,2)),

#     layers.Conv2D(32,3,3),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(pool_size=(2,2)),

#     layers.Conv2D(64,3,3),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(pool_size=(2,2)),

#     layers.Flatten(),
#     layers.Dense(64),
#     layers.Activation('relu'),
#     layers.Dropout(0.5),
#     layers.Dense(4),
#     layers.Activation('sigmoid')
# ])
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# epoch=100

# model.fit(x_data,y_data,epochs=epoch)

# models.save_model(model,'modelo.keras')

model=models.load_model('modelo.keras')

imagen=cv2.imread("C:/Users/pedro/Desktop/IA-CLASIFICADOR/IMAGENES/test/00001429_004.png")
imagen=cv2.resize(imagen,(width,height))

result=model.predict(np.array([imagen]))[0]

if result.argmax()==0:
    print("pneumo")
elif result.argmax()==1:
    print("NOFINDING")
elif result.argmax()==2:
    print("INFILTRATION")
elif result.argmax()==3:
    print("EFFUSION")