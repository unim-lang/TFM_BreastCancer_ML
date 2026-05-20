import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)


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


# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTamaño de los conjuntos:")
print("Train:", X_train.shape[0])
print("Test:", X_test.shape[0])

print("\nDistribución en test:")
print(y_test.value_counts())


# Modelos a evaluar
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


# Evaluar modelos en test
results = []
predictions = {}

print("\nEvaluando modelos en test...")

for model_name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Especificidad
    specificity = tn / (tn + fp)

    # Guardar métricas
    results.append({
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall_sensibilidad": recall_score(y_test, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    # Guardar predicciones para usar luego con el mejor modelo
    predictions[model_name] = {
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    print("\nModelo:", model_name)
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benigno", "Maligno"],
        zero_division=0
    ))


# Crear tabla de resultados
results_df = pd.DataFrame(results)

# Ordenamos priorizando:
# 1. Menos falsos negativos
# 2. Mayor sensibilidad
# 3. Mayor F1
# 4. Mayor ROC-AUC
results_df = results_df.sort_values(
    by=["fn", "recall_sensibilidad", "f1", "roc_auc"],
    ascending=[True, False, False, False]
)

print("\nResultados en test:")
print(results_df.round(4))


# Guardar resultados
results_df.to_csv("outputs/tables/resultados_modelos_test.csv", index=False)

print("\nArchivo guardado en: outputs/tables/resultados_modelos_test.csv")


# Seleccionar el mejor modelo según el orden anterior
best_model_name = results_df.iloc[0]["modelo"]

print("\nModelo seleccionado:", best_model_name)


# Recuperar predicciones del mejor modelo
y_pred_best = predictions[best_model_name]["y_pred"]
y_proba_best = predictions[best_model_name]["y_proba"]


# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benigno", "Maligno"],
    yticklabels=["Benigno", "Maligno"]
)

plt.title("Matriz de confusión - " + best_model_name)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.tight_layout()

plt.savefig("outputs/figures/matriz_confusion_mejor_modelo.png", dpi=300)
plt.close()

print("Matriz de confusión guardada en: outputs/figures/matriz_confusion_mejor_modelo.png")


# Curva ROC del mejor modelo
RocCurveDisplay.from_predictions(
    y_test,
    y_proba_best,
    name=best_model_name
)

plt.title("Curva ROC - " + best_model_name)
plt.tight_layout()

plt.savefig("outputs/figures/curva_roc_mejor_modelo.png", dpi=300)
plt.close()

print("Curva ROC guardada en: outputs/figures/curva_roc_mejor_modelo.png")