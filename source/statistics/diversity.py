import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot(histories: dict) -> None:
    fitness = _compute_fitness_per_run(histories)
    _plot_diversity(fitness)


def _compute_fitness_per_run(histories_dict: dict) -> list:
    fitness = []
    for key, histories in histories_dict.items():
        for history in histories:
            for idx_gen, generation in enumerate(history):
                for solution in generation:
                    fitness.append({
                        "key": key,
                        "generation": idx_gen,
                        "value": solution.f1
                    })
    return fitness


def _plot_diversity(fitness: list) -> None:

    df = pd.DataFrame(fitness)
    minimum = min(df["generation"])
    maximum = max(df["generation"])
    generations = np.linspace(minimum, maximum, 6, dtype=int)[1:]
    df_selected = df[df["generation"].isin(generations)]

    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df_selected, x="generation", y="value", hue="key")
    plt.legend(loc="lower left", prop={"size": 12})
    plt.ylabel("F1", size=14, labelpad=10)
    plt.xlabel("Generation", size=14, labelpad=10)
    plt.grid(True)
    plt.show()
