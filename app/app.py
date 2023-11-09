from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
app = Flask(__name__)
model = None  # Variable para almacenar el modelo de IA

# Ruta para cargar la página principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verifica si se ha enviado un archivo
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # Verifica si el archivo tiene un nombre y es una imagen
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Guarda la imagen en la carpeta 'uploads' en la ruta estática
            file_path = 'static/uploads/' + file.filename
            file.save(file_path)

            # Realiza la predicción utilizando el modelo de IA
            result = predict_image(file_path)

            return render_template('index.html', result=result)

    return render_template('index.html', result=None)

# Función para realizar la predicción en una imagen
def predict_image(image_path):
    # Carga la imagen y realiza preprocesamiento si es necesario
    img = cv2.imread(image_path)
    img = cv2.resize(img,(264, 264))  # Ajusta el tamaño según lo entrenado

    # Realiza la predicción utilizando el modelo
    result = model.predict(np.array([img]))[0]

    # Decodifica el resultado y obtén la etiqueta
    labels = ["Pneumothorax", "NoFinding", "Infiltration", "Nodule", "Pleural_Thickening", "Effusion"]
    predicted_label = labels[np.argmax(result)]

    return f'La IA predice que la imagen pertenece a la categoría: {predicted_label}'

if __name__ == '__main__':
    
    # custom_objects = {'Adam': tf.keras.optimizers.Adam}
    #custom_objects=custom_objects
    model = tf.keras.models.load_model('/Users/pedrord/Documents/GitHub/IA-CLASIFICADOR/app/modelo.h5')
    app.run(debug=True, port=5002)

    #250 8magense