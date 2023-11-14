from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
model = None  # Variable para almacenar el modelo de IA

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

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
            upload_folder = 'app/static/uploads/'

            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Realiza la predicción utilizando el modelo de IA
            result, image_filename = predict_image(file_path)

            # Construye la ruta completa a la imagen para la plantilla HTML
            image_path = os.path.join('uploads', image_filename)

    return render_template('index.html', result=result, image_path=image_path)


# Función para realizar la predicción en una imagen
def predict_image(image_path):
    # Carga la imagen y realiza preprocesamiento si es necesario
    img = cv2.imread(image_path)
    if img is not None:
        # Convertir imágenes en escala de grises a imágenes en color
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (264, 264))

    # Realiza la predicción utilizando el modelo
    result = model.predict(np.array([img]))[0]

    # # Decodifica el resultado y obtén la etiqueta
    # labels = ["Pneumothorax", "NoFinding", "Infiltration", "Nodule", "Pleural_Thickening", "Effusion"]
    # predicted_label = labels[np.argmax(result)]

    if result.argmax()==0:
        predicted_label=("pneumo")
    elif result.argmax()==1:
        predicted_label=("pleural")
    elif result.argmax()==2:
        predicted_label=("nofinding")  
    elif result.argmax()==3: 
        predicted_label=("NODULE")
    elif result.argmax()==4:
        predicted_label=("INFILTRATION")
    elif result.argmax()==5:
        predicted_label=("EFFUSION")        
    return f'Se predice que padece de: {predicted_label}', image_path


if __name__ == '__main__':
    # Carga el modelo de IA
    model = tf.keras.models.load_model('app/modelo.h5')
    app.run(debug=True, port=5002)
