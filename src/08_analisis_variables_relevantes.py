# Importa herramientas necesarias:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa herramientas de la librería scikit-learn:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importa los modelos que se usan en test:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Semilla de reproducibilidad
RANDOM_STATE = 29

# Cargar dataset limpio
df = pd.read_csv("data/wdbc_clean.csv")

print("Dataset cargado correctamente")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# Separar variables predictoras y variable objetivo
X = df.drop(columns=["id", "diagnosis"])

# B = benigno -> 0
# M = maligno -> 1
y = df["diagnosis"].map({"B": 0, "M": 1})

# Guardamos los nombres de las variables
feature_names = X.columns


# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

### 1. Interpretabilidad de Regresión Logística


logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

# Entrenar modelo
logistic_pipeline.fit(X_train, y_train)

# Extraer el modelo ya entrenado del pipeline
logistic_model = logistic_pipeline.named_steps["model"]

# Crear tabla de coeficientes
coeficientes = pd.DataFrame({
    "variable": feature_names,
    "coeficiente": logistic_model.coef_[0]
})

# Añadir valor absoluto para ordenar por importancia
coeficientes["abs_coeficiente"] = coeficientes["coeficiente"].abs()

# Ordenar de mayor a menor importancia
coeficientes = coeficientes.sort_values(
    by="abs_coeficiente",
    ascending=False
)

# Guardar tabla
coeficientes.to_csv(
    "outputs/tables/coeficientes_logistic_regression.csv",
    index=False
)

print("\nTop 10 variables según Regresión Logística:")
print(coeficientes.head(10))


# Gráfico de los 10 coeficientes más importantes
top_coef = coeficientes.head(10)
top_coef = top_coef.sort_values(by="coeficiente")

plt.figure(figsize=(9, 6))

sns.barplot(
    data=top_coef,
    x="coeficiente",
    y="variable"
)

plt.axvline(0, color="black", linewidth=1)
plt.title("Top 10 coeficientes - Regresión Logística optimizada")
plt.xlabel("Coeficiente")
plt.ylabel("Variable")
plt.tight_layout()

plt.savefig(
    "outputs/figures/coeficientes_logistic_regression_top10.png",
    dpi=300
)

plt.close()

print("\nCoeficientes guardados en: outputs/tables/coeficientes_logistic_regression.csv")
print("Figura guardada en: outputs/figures/coeficientes_logistic_regression_top10.png")



### 2. Interpretabilidad de Random Forest

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=100,
        random_state=RANDOM_STATE
    ))
])

# Entrenar modelo
rf_pipeline.fit(X_train, y_train)

# Extraer el modelo ya entrenado del pipeline
rf_model = rf_pipeline.named_steps["model"]

# Crear tabla de importancias
importancias = pd.DataFrame({
    "variable": feature_names,
    "importancia": rf_model.feature_importances_
})

# Ordenar por importancia
importancias = importancias.sort_values(
    by="importancia",
    ascending=False
)

# Guardar tabla
importancias.to_csv(
    "outputs/tables/importancia_variables_random_forest.csv",
    index=False
)

print("\nTop 10 variables según Random Forest:")
print(importancias.head(10))


# Gráfico de las 10 variables más importantes
top_importancias = importancias.head(10)
top_importancias = top_importancias.sort_values(by="importancia")

plt.figure(figsize=(9, 6))

sns.barplot(
    data=top_importancias,
    x="importancia",
    y="variable"
)

plt.title("Top 10 variables importantes - Random Forest optimizado")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()

plt.savefig(
    "outputs/figures/importancia_random_forest_top10.png",
    dpi=300
)

plt.close()

print("\nImportancias guardadas en: outputs/tables/importancia_variables_random_forest.csv")
print("Figura guardada en: outputs/figures/importancia_random_forest_top10.png")