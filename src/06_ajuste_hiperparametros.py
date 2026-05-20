# Importa herramientas necesarias:
import pandas as pd

# Importa herramientas de scikit-learn:
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importa los modelos que se usan en test:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Semilla de reproducibilidad
RANDOM_STATE = 29

# Métrica usada para seleccionar la mejor combinación de hiperparámetros
SELECTION_METRIC = "roc_auc"


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
# El ajuste se hace solo con train.
# El test queda reservado para la evaluación final.
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


# Validación cruzada interna para el ajuste
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Modelos e hiperparámetros que se van a probar
search_spaces = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"]
        }
    },

    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1]
        }
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }
    },

    "Gradient Boosting": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
            "model__subsample": [0.8, 1.0]
        }
    }
}


# Ejecutar GridSearchCV (prueba combinaciones de hiperparámetros)
results = []

print("\nIniciando ajuste de hiperparámetros...")

for model_name, config in search_spaces.items():

    print("\nAjustando modelo:", model_name)

    grid_search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        scoring=SELECTION_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    results.append({
        "modelo": model_name,
        "best_score_cv_roc_auc": grid_search.best_score_,
        "best_params": grid_search.best_params_
    })

    print("Mejor ROC-AUC CV:", round(grid_search.best_score_, 4))
    print("Mejores parámetros:", grid_search.best_params_)

    # Guardar resultados completos de cada GridSearch
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    file_name = model_name.replace(" ", "_").lower()
    output_path = "outputs/tables/gridsearch_" + file_name + ".csv"

    cv_results_df.to_csv(output_path, index=False)

    print("Resultados completos guardados en:", output_path)


# Crear tabla resumen
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="best_score_cv_roc_auc",
    ascending=False
)

results_df["best_score_cv_roc_auc"] = results_df["best_score_cv_roc_auc"].round(4)

print("\nResumen del ajuste de hiperparámetros:")
print(results_df)


# Guardar resumen
results_df.to_csv(
    "outputs/tables/resultados_ajuste_hiperparametros.csv",
    index=False
)

print("\nArchivo guardado en: outputs/tables/resultados_ajuste_hiperparametros.csv")