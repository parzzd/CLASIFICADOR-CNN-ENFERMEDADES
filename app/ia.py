import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

width = 264
height = 264

rut_entrada = "app\IMAGENES\entrada"
rut_prueba = "app\IMAGENES\test"

entrenamiento_x = []
entrenamiento_y = []

# for clase in os.listdir(rut_entrada):
#     clase_path = os.path.join(rut_entrada, clase)
#     if os.path.isdir(clase_path):  # Asegurarse de que es un directorio
#         for imagen_nombre in os.listdir(clase_path):
#             img_path = os.path.join(clase_path, imagen_nombre)
#             img = cv2.imread(img_path)
#             if img is not None:
#                 # Convertir imágenes en escala de grises a imágenes en color
#                 if len(img.shape) == 2:
#                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#                 resized_image = cv2.resize(img, (width, height))
#                 entrenamiento_x.append(resized_image)
#                 # Construir las etiquetas para cada imagen
#                 etiqueta = [0] * 6
#                 if clase == "Effusion":
#                     etiqueta[5] = 1
#                 elif clase == "Infiltration":
#                     etiqueta[4] = 1
#                 elif clase == "Nodule":
#                     etiqueta[3] = 1
#                 elif clase == "NoFinding":
#                     etiqueta[2] = 1
#                 elif clase == "Pleural_Thickening":
#                     etiqueta[1] = 1
#                 elif clase == "Pneumothorax":
#                     etiqueta[0] = 1
#                 entrenamiento_y.append(etiqueta)
#             else:
#                 print(f"Error al cargar la imagen: {img_path}")

# x_data = np.array(entrenamiento_x)
# y_data = np.array(entrenamiento_y)

# # Dividir los datos en conjuntos de entrenamiento y prueba
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# model = tf.keras.Sequential([
#     layers.Conv2D(64, 3, 3, input_shape=(width, height, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),

#     layers.Conv2D(128, 3, 3, activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),

#     layers.Conv2D(256, 3, 3, activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),

#     layers.Dropout(0.5),
#     layers.Flatten(),
#     layers.Dense(6, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# epoch = 100

# model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))

# # Guardar el modelo en una ruta relativa o absoluta
# model.save('modelo.h5')

model = tf.keras.models.load_model('modelo.h5')

imagen=cv2.imread("D:/CLASIFICADOR-CNN-ENFERMEDADES/app/IMAGENES/test/00001373_038.png")
imagen=cv2.resize(imagen,(width,height))

result=model.predict(np.array([imagen]))[0]

if result.argmax()==0:
    print("pneumo")
elif result.argmax()==1:
    print("pleural")
elif result.argmax()==2:
    print("nofinding")  
elif result.argmax()==3: 
    print("NODUEL")
elif result.argmax()==4:
    print("INFILTRATION")
elif result.argmax()==5:
    print("EFFUSION") 