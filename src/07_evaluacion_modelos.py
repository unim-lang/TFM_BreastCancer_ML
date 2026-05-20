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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importa las métricas:
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
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

print("\nEvaluación final de modelos optimizados")
print("Train:", X_train.shape[0])
print("Test:", X_test.shape[0])

print("\nDistribución en test:")
print(y_test.value_counts())


# Modelos optimizados según la fase de ajuste de hiperparámetros
models = {
    "Logistic Regression Optimized": LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE
    ),

    "SVM Optimized": SVC(
        C=10,
        gamma=0.01,
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ),

    "Random Forest Optimized": RandomForestClassifier(
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=100,
        random_state=RANDOM_STATE
    ),

    "Gradient Boosting Optimized": GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=1.0,
        random_state=RANDOM_STATE
    ),
}


# Evaluar modelos
results = []
predictions = {}

for model_name, model in models.items():

    print("\nModelo:", model_name)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predecir clase y probabilidad de maligno
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Especificidad
    specificity = tn / (tn + fp)

    # Métricas
    results.append({
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall_sensibilidad": recall_score(y_test, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    # Guardar predicciones para las figuras
    predictions[model_name] = {
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benigno", "Maligno"]
    ))

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("Falsos positivos:", fp)
    print("Falsos negativos:", fn)


# Crear tabla final
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

results_clean = results_df.round(4)

print("\nTabla final de modelos optimizados:")
print(results_clean)


# Guardar tabla final
results_clean.to_csv(
    "outputs/tables/resultados_finales_modelos_optimizados.csv",
    index=False
)

print("\nArchivo guardado en: outputs/tables/resultados_finales_modelos_optimizados.csv")


# Guardar matriz de confusión de cada modelo
for model_name, values in predictions.items():

    y_pred = values["y_pred"]

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"]
    )

    plt.title("Matriz de confusión - " + model_name)
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").lower()
    output_path = "outputs/figures/matriz_confusion_" + safe_name + ".png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Matriz guardada:", output_path)


# Curvas ROC comparativas
plt.figure(figsize=(8, 6))
ax = plt.gca()

for model_name, values in predictions.items():

    RocCurveDisplay.from_predictions(
        y_test,
        values["y_proba"],
        name=model_name,
        ax=ax
    )

plt.title("Curvas ROC - Modelos optimizados")
plt.tight_layout()

plt.savefig("outputs/figures/curvas_roc_modelos_optimizados.png", dpi=300)
plt.close()

print("Curvas ROC guardadas en: outputs/figures/curvas_roc_modelos_optimizados.png")


# Gráfico comparativo de métricas finales
metrics_to_plot = [
    "accuracy",
    "precision",
    "recall_sensibilidad",
    "specificity",
    "f1",
    "roc_auc"
]

long_df = results_clean.melt(
    id_vars="modelo",
    value_vars=metrics_to_plot,
    var_name="metrica",
    value_name="valor"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=long_df,
    x="modelo",
    y="valor",
    hue="metrica"
)

plt.title("Comparativa final de modelos optimizados")
plt.xlabel("Modelo")
plt.ylabel("Valor")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("outputs/figures/comparativa_final_modelos_optimizados.png", dpi=300)
plt.close()

print("Comparativa final guardada.")


# Gráfico de falsos positivos y falsos negativos
error_df = results_clean[["modelo", "fp", "fn"]].melt(
    id_vars="modelo",
    value_vars=["fp", "fn"],
    var_name="tipo_error",
    value_name="numero_errores"
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=error_df,
    x="modelo",
    y="numero_errores",
    hue="tipo_error"
)

plt.title("Falsos positivos y falsos negativos por modelo optimizado")
plt.xlabel("Modelo")
plt.ylabel("Número de errores")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/figures/errores_fp_fn_modelos_optimizados.png", dpi=300)
plt.close()

print("Gráfico de errores guardado.")