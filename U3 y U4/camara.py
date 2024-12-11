import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('modelo_alimentos.h5')

# clases
clases = [
    'Alimentos en conserva', 'Bebidas procesadas', 'Carnes procesadas', 'Comida lista para comer o precocinada',
    'Cereales procesados', 'Panaderia y pasteleria', 'Lacteos procesados', 'Snacks ultraprocesados'
]

# preprocesar la imagen
def preprocesar_imagen(frame, tamaño=(150, 150)):
    img_resized = cv2.resize(frame, tamaño)
    img_normalizada = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalizada, axis=0)
    return img_expanded

# cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # frame en tiempo real
    cv2.imshow("Cámara - Snack", frame)

    # frame para la predicción
    img_preprocesada = preprocesar_imagen(frame)

    # Realizar la predicción
    pred = model.predict(img_preprocesada)
    clase_predicha = np.argmax(pred, axis=1)[0]
    clase_nombre = clases[clase_predicha]

    # predicción en el frame
    texto = f"Tipo de Snack: {clase_nombre}"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # frame con la predicciónqq
    cv2.imshow("Cámara - Snack", frame)

    # Salir 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()