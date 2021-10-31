import numpy as np
from scipy.stats import chi2, rankdata
from typing import List, Tuple


def test(histories_a: List[list], histories_b: List[list]) -> Tuple[float, float]:
    """A - B"""
    cut_points = _get_cut_points(histories_a[0])
    page_l = _get_page_l(histories_a, histories_b, cut_points)
    p_value = _get_p_value(page_l, len(histories_a), len(cut_points))
    return page_l, p_value


def _get_cut_points(history) -> List[int]:
    N_CUT_POINTS = 10
    length = len(history) // N_CUT_POINTS
    cut_points = np.arange(0, len(history), length)
    return cut_points.tolist()


def _get_page_l(histories_a: List[list], histories_b: List[list], cut_points: List[int]) -> float:
    rank_lists = []
    for history_a, history_b in zip(histories_a, histories_b):

        difference = []
        for cut_point in cut_points:
            difference.append(
                _get_best_fitness(history_b[cut_point]) - _get_best_fitness(history_a[cut_point])
            )

        rank = rankdata(difference)
        rank_lists.append(rank)

    rank_sum = []
    for i in range(len(rank_lists[0])):
        rank_sum.append(
            sum([row[i] for row in rank_lists]) * (i + 1)
        )
    
    return sum(rank_sum)


def _get_p_value(page_l: float, n_runs: int, n_cut_points: int) -> float:
    chi_squared = (
        ((12.0 * page_l - 3.0 * n_runs * n_cut_points * (n_cut_points+1.0)**2.0) ** 2.0) /
        ((n_runs * n_cut_points**2.0 * (n_cut_points**2.0 - 1.0) * (n_cut_points + 1.0)))
    )
    p_two_tailed = 1 - chi2.cdf(chi_squared, 1)
    p_value = p_two_tailed / 2.0
    return p_value


def _get_best_fitness(generation: List[tuple]) -> List[float]:
    for solution in generation:
        best_fitness = float("-inf")
        fitness = solution[1].f1
        if fitness > best_fitness:
            best_fitness = fitness
    return best_fitness
