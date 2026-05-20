# Importa las herramientas necesarias:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa herramientas de la librería scikit-learn:
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importa los modelos que se usan en test:
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

print("\nDistribución de clases:")
print(y.value_counts())


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


# Validación cruzada repetida
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=RANDOM_STATE
)


# Métricas
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall_sensibilidad": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# Listas para guardar resultados
results_summary = []
results_all_folds = []

print("\nEjecutando validación cruzada robusta...")
print("5 folds x 10 repeticiones = 50 evaluaciones por modelo")


# Evaluar cada modelo
for model_name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    # Resumen por modelo
    results_summary.append({
        "modelo": model_name,

        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),

        "precision_mean": scores["test_precision"].mean(),
        "precision_std": scores["test_precision"].std(),

        "recall_sensibilidad_mean": scores["test_recall_sensibilidad"].mean(),
        "recall_sensibilidad_std": scores["test_recall_sensibilidad"].std(),

        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),

        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
    })

    # Resultados de cada fold/repetición
    for i in range(len(scores["test_accuracy"])):

        results_all_folds.append({
            "modelo": model_name,
            "fold_repetition": i + 1,
            "accuracy": scores["test_accuracy"][i],
            "precision": scores["test_precision"][i],
            "recall_sensibilidad": scores["test_recall_sensibilidad"][i],
            "f1": scores["test_f1"][i],
            "roc_auc": scores["test_roc_auc"][i],
        })

    print(model_name, "completado")


# Crear DataFrames
summary_df = pd.DataFrame(results_summary)
all_folds_df = pd.DataFrame(results_all_folds)

# Ordenar resumen por ROC-AUC medio
summary_df = summary_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

# Guardar resultados
summary_df.round(4).to_csv(
    "outputs/tables/resultados_validacion_robusta_resumen.csv",
    index=False
)

all_folds_df.to_csv(
    "outputs/tables/resultados_validacion_robusta_folds.csv",
    index=False
)

print("\nResumen de validación robusta:")
print(summary_df.round(4))

print("\nArchivos guardados:")
print("outputs/tables/resultados_validacion_robusta_resumen.csv")
print("outputs/tables/resultados_validacion_robusta_folds.csv")


# Configuración visual
sns.set_theme(style="whitegrid")


# Gráfico de ROC-AUC medio
plot_df = summary_df.sort_values(by="roc_auc_mean", ascending=False)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=plot_df,
    x="roc_auc_mean",
    y="modelo"
)

plt.title("ROC-AUC medio en validación cruzada robusta")
plt.xlabel("ROC-AUC medio")
plt.ylabel("Modelo")
plt.xlim(0, 1.05)
plt.tight_layout()

plt.savefig("outputs/figures/validacion_robusta_roc_auc_medio.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/validacion_robusta_roc_auc_medio.png")


# Boxplot de ROC-AUC
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=all_folds_df,
    x="modelo",
    y="roc_auc"
)

plt.title("Distribución del ROC-AUC por modelo en validación robusta")
plt.xlabel("Modelo")
plt.ylabel("ROC-AUC")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/figures/validacion_robusta_boxplot_roc_auc.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/validacion_robusta_boxplot_roc_auc.png")


# Boxplot de sensibilidad
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=all_folds_df,
    x="modelo",
    y="recall_sensibilidad"
)

plt.title("Distribución de la sensibilidad por modelo en validación robusta")
plt.xlabel("Modelo")
plt.ylabel("Sensibilidad")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/figures/validacion_robusta_boxplot_sensibilidad.png", dpi=300)
plt.close()

print("Figura guardada: outputs/figures/validacion_robusta_boxplot_sensibilidad.png")