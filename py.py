import os
import cv2
import mediapipe as mp
import pandas as pd

# Ruta donde están las carpetas A, B, C...
DATASET_DIR = "A"
OUTPUT_CSV = "landmarks.csv"

mp_hands = mp.solutions.hands

# Aquí guardaremos todos los datos
data = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    # Recorre cada carpeta (A, B, C...)
    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Procesando letra: {label}")

        # Recorre cada imagen dentro de la carpeta
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # Leer imagen
            image = cv2.imread(img_path)
            if image is None:
                print(f"Imagen inválida: {img_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Si no detecta mano, saltar
            if not results.multi_hand_landmarks:
                print(f"Sin mano detectada: {img_name}")
                continue

            # Extraer landmarks
            hand = results.multi_hand_landmarks[0]
            row = []

            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Agregar etiqueta (A, B, C…)
            row.append(label)

            data.append(row)

# Crear columnas
columns = [f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
columns.append("label")

# Crear DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("\n-----------------------------------")
print("¡Listo! CSV generado correctamente.")
print("Archivo:", OUTPUT_CSV)
print("-----------------------------------")