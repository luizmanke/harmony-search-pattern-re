from scipy.stats import ttest_ind
from typing import List


def test(histories_a: List[list], histories_b: List[list]) -> float:
    best_fitness_a = _get_best_fitness(histories_a)
    best_fitness_b = _get_best_fitness(histories_b)
    _, p_value = ttest_ind(best_fitness_a, best_fitness_b)
    return p_value


def _get_best_fitness(histories: List[list]) -> List[float]:
    best_fitness_list = []
    for history in histories:
        best_fitness = float("-inf")
        for harmony in history[-1]:
            fitness = harmony[1].f1
            if fitness > best_fitness:
                best_fitness = fitness
        best_fitness_list.append(best_fitness)
    return best_fitness_list
