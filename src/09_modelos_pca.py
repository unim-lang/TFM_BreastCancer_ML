# Importa herramientas necesarias:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa herramientas de la librería scikit-learn:
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Importa los modelos que se usan en test:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
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

print("\nModelos con PCA")
print("Train:", X_train.shape[0])
print("Test:", X_test.shape[0])
print("Variables originales:", X.shape[1])



# 1. PCA exploratorio

# Estandarizar solo el conjunto de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Aplicar PCA con todos los componentes
pca = PCA()
pca.fit(X_train_scaled)

# Varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = varianza_explicada.cumsum()

pca_variance_df = pd.DataFrame({
    "componente": ["PC" + str(i + 1) for i in range(len(varianza_explicada))],
    "varianza_explicada": varianza_explicada,
    "varianza_acumulada": varianza_acumulada
})

# Guardar tabla de varianza explicada
pca_variance_df.to_csv(
    "outputs/tables/pca_varianza_explicada.csv",
    index=False
)

print("\nVarianza explicada por los primeros componentes:")
print(pca_variance_df.head(10).round(4))


# Número de componentes necesarios para llegar al 95% de varianza
n_components_95 = (varianza_acumulada >= 0.95).argmax() + 1

print("\nComponentes necesarios para explicar al menos el 95% de la varianza:", n_components_95)


# Gráfico de varianza acumulada
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(varianza_acumulada) + 1),
    varianza_acumulada,
    marker="o"
)

plt.axhline(0.95, linestyle="--", label="95% varianza")
plt.axvline(n_components_95, linestyle="--", label=str(n_components_95) + " componentes")

plt.title("Varianza explicada acumulada por PCA")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig("outputs/figures/pca_varianza_acumulada.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/pca_varianza_acumulada.png")



# 2. Visualización con PC1 y PC2

pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)

pca_2d_df = pd.DataFrame({
    "PC1": X_train_pca_2d[:, 0],
    "PC2": X_train_pca_2d[:, 1],
    "diagnosis": y_train.map({0: "Benigno", 1: "Maligno"}).values
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_2d_df,
    x="PC1",
    y="PC2",
    hue="diagnosis",
    alpha=0.8
)

plt.title("Visualización PCA: PC1 vs PC2")
plt.xlabel("PC1 (" + str(round(pca_2d.explained_variance_ratio_[0] * 100, 2)) + "% varianza)")
plt.ylabel("PC2 (" + str(round(pca_2d.explained_variance_ratio_[1] * 100, 2)) + "% varianza)")
plt.tight_layout()

plt.savefig("outputs/figures/pca_pc1_pc2.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/pca_pc1_pc2.png")



# 3. Modelos con PCA

models_pca = {
    "Logistic Regression + PCA": LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE
    ),

    "SVM + PCA": SVC(
        C=10,
        gamma=0.01,
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ),

    "Random Forest + PCA": RandomForestClassifier(
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=100,
        random_state=RANDOM_STATE
    ),

    "Gradient Boosting + PCA": GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=1.0,
        random_state=RANDOM_STATE
    ),
}


# Validación cruzada
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Métricas
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall_sensibilidad": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


cv_results = []
test_results = []

print("\nEvaluando modelos con PCA...")


for model_name, model in models_pca.items():

    print("Modelo:", model_name)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components_95)),
        ("model", model)
    ])

    # Validación cruzada sobre train
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    cv_results.append({
        "modelo": model_name,
        "n_componentes_pca": n_components_95,

        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),

        "precision_mean": scores["test_precision"].mean(),
        "precision_std": scores["test_precision"].std(),

        "recall_sensibilidad_mean": scores["test_recall_sensibilidad"].mean(),
        "recall_sensibilidad_std": scores["test_recall_sensibilidad"].std(),

        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),

        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std()
    })

    # Evaluación en test
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    test_results.append({
        "modelo": model_name,
        "n_componentes_pca": n_components_95,
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


# Crear tablas de resultados
cv_results_df = pd.DataFrame(cv_results)
test_results_df = pd.DataFrame(test_results)

cv_results_df = cv_results_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

test_results_df = test_results_df.sort_values(
    by=["fn", "recall_sensibilidad", "f1", "roc_auc"],
    ascending=[True, False, False, False]
)

cv_results_clean = cv_results_df.round(4)
test_results_clean = test_results_df.round(4)


# Guardar resultados
cv_results_clean.to_csv(
    "outputs/tables/resultados_modelos_pca_cv.csv",
    index=False
)

test_results_clean.to_csv(
    "outputs/tables/resultados_modelos_pca_test.csv",
    index=False
)

print("\nResultados test con PCA:")
print(test_results_clean)

print("\nArchivos guardados:")
print("outputs/tables/resultados_modelos_pca_cv.csv")
print("outputs/tables/resultados_modelos_pca_test.csv")


 
# 4. Gráfico comparativo de modelos con PCA
 
metrics_to_plot = [
    "accuracy",
    "precision",
    "recall_sensibilidad",
    "specificity",
    "f1",
    "roc_auc"
]

long_df = test_results_clean.melt(
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

plt.title("Comparativa de modelos con PCA en test")
plt.xlabel("Modelo")
plt.ylabel("Valor")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("outputs/figures/comparativa_modelos_pca_test.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/comparativa_modelos_pca_test.png")


 
# 5. Comparación sin PCA vs con PCA

# Esta parte solo se ejecuta si ya existe la tabla de modelos optimizados sin PCA
try:
    without_pca_df = pd.read_csv("outputs/tables/resultados_finales_modelos_optimizados.csv")

    without_pca_df["estrategia"] = "Sin PCA"
    test_results_clean["estrategia"] = "Con PCA"

    # Crear nombres comparables
    without_pca_df["modelo_base"] = without_pca_df["modelo"].str.replace(" Optimized", "", regex=False)
    test_results_clean["modelo_base"] = test_results_clean["modelo"].str.replace(" + PCA", "", regex=False)

    comparison_df = pd.concat([
        without_pca_df[[
            "modelo_base",
            "estrategia",
            "accuracy",
            "precision",
            "recall_sensibilidad",
            "specificity",
            "f1",
            "roc_auc",
            "fp",
            "fn"
        ]],
        test_results_clean[[
            "modelo_base",
            "estrategia",
            "accuracy",
            "precision",
            "recall_sensibilidad",
            "specificity",
            "f1",
            "roc_auc",
            "fp",
            "fn"
        ]]
    ])

    comparison_df.to_csv(
        "outputs/tables/comparacion_sin_pca_vs_con_pca.csv",
        index=False
    )

    print("Archivo guardado: outputs/tables/comparacion_sin_pca_vs_con_pca.csv")


    # Gráfico comparativo de F1
    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=comparison_df,
        x="modelo_base",
        y="f1",
        hue="estrategia"
    )

    plt.title("Comparación F1: modelos sin PCA vs con PCA")
    plt.xlabel("Modelo")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig("outputs/figures/comparacion_f1_sin_pca_vs_con_pca.png", dpi=300)
    plt.close()

    print("Figura guardada: outputs/figures/comparacion_f1_sin_pca_vs_con_pca.png")


    # Gráfico comparativo de falsos negativos
    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=comparison_df,
        x="modelo_base",
        y="fn",
        hue="estrategia"
    )

    plt.title("Comparación de falsos negativos: sin PCA vs con PCA")
    plt.xlabel("Modelo")
    plt.ylabel("Falsos negativos")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig("outputs/figures/comparacion_fn_sin_pca_vs_con_pca.png", dpi=300)
    plt.close()

    print("Figura guardada: outputs/figures/comparacion_fn_sin_pca_vs_con_pca.png")

except FileNotFoundError:
    print("\nNo se encontró resultados_finales_modelos_optimizados.csv.")
    print("No se ha generado comparación sin PCA vs con PCA.")