import zipfile
import pandas as pd


# Ruta del archivo ZIP
zip_path = "data/breast+cancer+wisconsin+diagnostic.zip"

# Nombres base de las características
base_features = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

# Crear la lista completa de columnas
columns = ["id", "diagnosis"]

for suffix in ["mean", "se", "worst"]:
    for feature in base_features:
        columns.append(feature + "_" + suffix)

# Leer el archivo wdbc.data que está dentro del ZIP
with zipfile.ZipFile(zip_path) as zip_file:
    with zip_file.open("wdbc.data") as file:
        df = pd.read_csv(file, header=None, names=columns)

# Comprobaciones
print("Dataset cargado correctamente")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nPrimeras filas:")
print(df.head())

print("\nDistribución de la variable objetivo:")
print(df["diagnosis"].value_counts())

print("\nValores nulos totales:")
print(df.isnull().sum().sum())

# Guardar CSV limpio
df.to_csv("data/wdbc_clean.csv", index=False)

print("\nArchivo guardado en: data/wdbc_clean.csv")