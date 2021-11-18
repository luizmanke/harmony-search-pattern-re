from scipy.stats import shapiro
from typing import List, Tuple


def test(histories: List[list]) -> Tuple[float, float]:
    best_fitness = _get_best_fitness(histories)
    w_statistic, p_value = shapiro(best_fitness)
    return round(w_statistic, 4), round(p_value, 4)


def _get_best_fitness(histories: List[list]) -> List[float]:
    best_fitness_list = []
    for history in histories:
        best_fitness = float("-inf")
        for harmony in history[-1]:
            fitness = harmony.f1
            if fitness > best_fitness:
                best_fitness = fitness
        best_fitness_list.append(best_fitness)
    return best_fitness_list
