import cv2
import mediapipe as mp
import pandas as pd
import os

# Carpeta donde están las imágenes
carpeta = "C"

# Nombre del archivo CSV de salida
archivo_salida = "landmarks_C.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lista donde guardaremos los landmarks
datos = []

# Encabezados
columnas = []
for i in range(21):
    columnas += [f"{i}_x", f"{i}_y", f"{i}_z"]
columnas.append("label")  # Etiqueta de la letra

# Procesar cada imagen en la carpeta
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):

            ruta = os.path.join(carpeta, archivo)
            img = cv2.imread(ruta)

            if img is None:
                print(f"Error al leer {archivo}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultado = hands.process(img_rgb)

            if not resultado.multi_hand_landmarks:
                print(f"No se detectó mano en {archivo}")
                continue

            # Procesar landmarks
            mano = resultado.multi_hand_landmarks[0]

            fila = []
            for lm in mano.landmark:
                fila.append(lm.x)
                fila.append(lm.y)
                fila.append(lm.z)

            fila.append("C")  # Cambia por la letra que estás capturando

            datos.append(fila)
            print(f"Procesado: {archivo}")

# Guardar CSV
df = pd.DataFrame(datos, columns=columnas)
df.to_csv(archivo_salida, index=False)

print("Listo. CSV generado:", archivo_salida)
