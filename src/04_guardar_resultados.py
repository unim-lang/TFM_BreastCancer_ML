# Importa las herramientas necesarias:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados anteriores
baseline_df = pd.read_csv("outputs/tables/resultados_modelos_baseline.csv")
test_df = pd.read_csv("outputs/tables/resultados_modelos_test.csv")

print("Resultados baseline cargados:")
print(baseline_df.shape)

print("\nResultados test cargados:")
print(test_df.shape)

print("\nModelos en test:")
print(test_df["modelo"].tolist())


# Redondear tablas para que queden más limpias
baseline_clean = baseline_df.round(4)
test_clean = test_df.round(4)

# Guardar tablas limpias
baseline_clean.to_csv(
    "outputs/tables/resultados_modelos_baseline_limpio.csv",
    index=False
)

test_clean.to_csv(
    "outputs/tables/resultados_modelos_test_limpio.csv",
    index=False
)

print("\nTablas limpias guardadas.")


# Configuración visual de los gráficos
sns.set_theme(style="whitegrid")


# Métricas que queremos representar
metrics = [
    "accuracy",
    "precision",
    "recall_sensibilidad",
    "specificity",
    "f1",
    "roc_auc",
]


# Crear un gráfico individual para cada métrica
for metric in metrics:

    plot_df = test_df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x=metric,
        y="modelo"
    )

    plt.title("Comparativa de modelos según " + metric)
    plt.xlabel(metric)
    plt.ylabel("Modelo")
    plt.xlim(0, 1.05)
    plt.tight_layout()

    output_path = "outputs/figures/comparativa_" + metric + "_test.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Figura guardada:", output_path)


# Gráfico conjunto de todas las métricas en test
long_df = test_df.melt(
    id_vars="modelo",
    value_vars=metrics,
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

plt.title("Comparativa conjunta de métricas en test")
plt.xlabel("Modelo")
plt.ylabel("Valor")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("outputs/figures/comparativa_conjunta_metricas_test.png", dpi=300)
plt.close()

print("Figura conjunta guardada.")


# Identificar el mejor modelo por cada métrica
best_by_metric = []

for metric in metrics:

    best_row = test_df.sort_values(by=metric, ascending=False).iloc[0]

    best_by_metric.append({
        "metrica": metric,
        "mejor_modelo": best_row["modelo"],
        "valor": best_row[metric]
    })


best_by_metric_df = pd.DataFrame(best_by_metric)
best_by_metric_df = best_by_metric_df.round(4)

best_by_metric_df.to_csv(
    "outputs/tables/mejores_modelos_por_metrica_test.csv",
    index=False
)

print("\nMejores modelos por métrica:")
print(best_by_metric_df)

print("\nArchivo guardado en: outputs/tables/mejores_modelos_por_metrica_test.csv")


# Mostrar tabla de test ordenada por ROC-AUC
print("\nResultados test ordenados por ROC-AUC:")
print(test_clean.sort_values(by="roc_auc", ascending=False))