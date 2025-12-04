import pandas as pd
import glob

# Cargar todos los CSV de la carpeta dataset/
archivos = glob.glob("dataset/*.csv")

df = pd.DataFrame()

for archivo in archivos:
    temp = pd.read_csv(archivo)
    df = pd.concat([df, temp], ignore_index=True)

print("Total de muestras:", len(df))
print(df.head())
