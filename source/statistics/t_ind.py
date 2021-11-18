from scipy.stats import ttest_ind
from typing import List, Tuple


def test(histories_a: List[list], histories_b: List[list]) -> Tuple[float, float]:
    best_fitness_a = _get_best_fitness(histories_a)
    best_fitness_b = _get_best_fitness(histories_b)
    w_statistic, p_value = ttest_ind(best_fitness_a, best_fitness_b)
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
