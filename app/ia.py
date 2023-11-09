import tensorflow as tf
from keras import layers,models
import os
import numpy as np
import cv2
#dimension 264x264
#pip install --upgrade tensorflow
#pip install --upgrade keras

#model.save('model.h5')
#loaded_model = tf.keras.models.load_model('model.h5')


width = 264
height = 264
rut_entrada = "app/imagenes/entrada/"
rut_prueba = "app/imagenes/test"

entrenamiento_x = []
entrenamiento_y = []

for i in os.listdir(rut_entrada):
    for j in os.listdir(rut_entrada + i):
        img = cv2.imread(rut_entrada + i + "/" + j)
        resized_image = cv2.resize(img, (width, height))

        entrenamiento_x.append(resized_image)
        if i == "Effusion":
            entrenamiento_y.append([0, 0, 0, 0, 0, 1])
        elif i == "Infiltration":
            entrenamiento_y.append([0, 0, 0, 0, 1, 0])
        elif i == "Nodule":
            entrenamiento_y.append([0, 0, 0, 1, 0, 0])
        elif i == "NoFinding":
            entrenamiento_y.append([0, 0, 1, 0, 0, 0])
        elif i == "Pleural_Thickening":
            entrenamiento_y.append([0, 1, 0, 0, 0, 0])
        elif i == "Pneumothorax":
            entrenamiento_y.append([1, 0, 0, 0, 0, 0])

x_data = np.array(entrenamiento_x)
y_data = np.array(entrenamiento_y)

model = tf.keras.Sequential([
    layers.Conv2D(64, 3, 3, input_shape=(width, height, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, 3, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(256, 3, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dropout(0.5),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


epoch = 100

model.fit(x_data, y_data, epochs=epoch)

model.save('modelo.h5') # Corregida la función de guardar el modelo









# model = tf.keras.models.load_model('/Users/pedrord/Documents/GitHub/IA-CLASIFICADOR/app/modelo.h5')

# imagen=cv2.imread("/Users/pedrord/Documents/GitHub/IA-CLASIFICADOR/app/imagenes/test/00001376_008.png")
# imagen=cv2.resize(imagen,(width,height))

# result=model.predict(np.array([imagen]))[0]

# if result.argmax()==0:
#     print("pneumo")
# elif result.argmax()==1:
#     print("pleural")
# elif result.argmax()==2:
#     print("nofinding")  
# elif result.argmax()==3:
#     print("NODUEL")
# elif result.argmax()==4:
#     print("INFILTRATION")
# elif result.argmax()==5:
#     print("EFFUSION")    dsd
    