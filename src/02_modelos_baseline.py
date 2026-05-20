import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


# Modelos baseline
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


# Validación cruzada estratificada
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Métricas de evaluación
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# Evaluación de modelos
results = []

print("\nEntrenando modelos con validación cruzada...")

for model_name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    results.append({
        "modelo": model_name,

        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),

        "precision_mean": scores["test_precision"].mean(),
        "precision_std": scores["test_precision"].std(),

        "recall_mean": scores["test_recall"].mean(),
        "recall_std": scores["test_recall"].std(),

        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),

        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
    })

    print(model_name, "completado")


# Crear tabla de resultados
results_df = pd.DataFrame(results)

# Ordenar por ROC-AUC medio
results_df = results_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

print("\nResultados baseline:")
print(results_df.round(4))


# Guardar resultados
results_df.to_csv("outputs/tables/resultados_modelos_baseline.csv", index=False)

print("\nArchivo guardado en: outputs/tables/resultados_modelos_baseline.csv")