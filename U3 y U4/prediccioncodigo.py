import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

clases = [
    'Alimentos en conserva', 'Bebidas procesadas', 'Carnes procesadas', 'Comidas listas para comer o precocinadas', 
    'Productos de cereales procesados', 'Productos de panadería y pastelería', 'Productos lácteos procesados', 'Snacks y alimentos ultraprocesados'
]
base_dir = 'C:/Users/Karla/Documents/U3y4/Dataset'

def cargar_imagenes(base_dir, clases, tamaño=(150, 150)):
    imagenes = []
    etiquetas = []
    for etiqueta, clase in enumerate(clases):
        clase_dir = os.path.join(base_dir, clase)
        for archivo in os.listdir(clase_dir):
            if archivo.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
                img_path = os.path.join(clase_dir, archivo)
                img = cv2.imread(img_path)
                if img is not None:  # Asegurarse de que la imagen se cargue correctamente
                    img_resized = cv2.resize(img, tamaño)
                    img_normalizada = img_resized / 255.0
                    imagenes.append(img_normalizada)
                    etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

imagenes, etiquetas = cargar_imagenes(base_dir, clases)

etiquetas_onehot = to_categorical(etiquetas, num_classes=len(clases))

X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas_onehot, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(clases), activation='softmax')  # Salida con tantas neuronas como clases
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test)
print(f'Pérdida en test: {score[0]}, Precisión en test: {score[1]}')

model.save('modelo_alimentos.h5')

def predecir_imagen(imagen_path):
    img = cv2.imread(imagen_path)
    if img is not None:
        img_resized = cv2.resize(img, (150, 150))
        img_normalizada = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalizada, axis=0)
        pred = model.predict(img_expanded)
        clase_predicha = np.argmax(pred, axis=1)[0]
        print(f"El alimento pertenece a la categoría: {clases[clase_predicha]}.")
    else:
        print(f"No se pudo cargar la imagen: {imagen_path}")

imagen_para_predecir = 'C:/Users/Karla/Documents/U3y4/Dataset/Bebidas procesadas/bebida_procesada-gatorade de Mandarina.jpg'
predecir_imagen(imagen_para_predecir)
