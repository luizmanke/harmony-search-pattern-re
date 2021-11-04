import matplotlib.pyplot as plt
import numpy as np


def plot(histories: dict) -> None:
    fitness = _compute_best_fitness_per_run(histories)
    _plot_hists(fitness)


def _compute_best_fitness_per_run(histories_dict: dict) -> dict:
    fitness_dict = {}
    for key, histories in histories_dict.items():
        fitness_dict[key] = []
        for history in histories:
            best_fitness = max([solution[1].f1 for solution in history[-1]])
            fitness_dict[key].append(best_fitness)
    return fitness_dict


def _plot_hists(fitness: dict) -> None:
    plt.figure(figsize=(16, 6))
    for key, values in fitness.items():
        plt.hist(values, label=key)
    plt.legend(loc="upper left", prop={"size": 12})
    plt.ylabel("f1", size=14, labelpad=10)
    plt.grid(True)
    plt.show()
