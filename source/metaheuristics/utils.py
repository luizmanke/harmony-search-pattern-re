# TODO: use SOLUTION class
# TODO: reuse mutation function
# TODO: update statistics



import numpy as np
import random
from copy import deepcopy
from abc import abstractmethod
from multiprocessing import Pool
from sklearn import metrics
from typing import Callable, List

ANY_VALUE = -100_000


class Fitness:

    def __init__(self, f1: float, precision: float, recall: float) -> None:
        self.f1 = f1
        self.precision = precision
        self.recall = recall

    def __gt__(self, other):
        return self.f1 > other

    def __lt__(self, other):
        return self.f1 < other


class Solution(Fitness):
    pass


class ObjectiveFunction:

    def get_fitness(
        self,
        samples: List[dict],
        labels: List[int],
        solution: List[dict]
    ) -> Fitness:
        predictions = []
        for sample in samples:
            is_positive = False
            for parameter in solution:
                if self._are_equal(sample["between"], parameter["between"]):
                    is_positive = True
                    break
            predictions.append(is_positive)
        return Fitness(
            metrics.f1_score(labels, predictions),
            metrics.precision_score(labels, predictions),
            metrics.recall_score(labels, predictions)
        )

    @staticmethod
    def _are_equal(current: list, expected: list) -> bool:
        are_equal = True
        if len(current) != len(expected):
            are_equal = False
        else:
            for item_1, item_2 in zip(current, expected):
                if item_1 != item_2:
                    if item_2 != -100_000:
                        are_equal = False
                        break
        return are_equal


class BaseMetaheuristic(ObjectiveFunction):

    def __init__(
        self,
        n_parameters: int,
        n_solutions: int,
        n_generations: int,
        n_runs: int
    ) -> None:
        self.n_parameters_ = n_parameters
        self.n_solutions_ = n_solutions
        self.n_generations_ = n_generations
        self.n_runs_ = n_runs

    @abstractmethod
    def run(self) -> dict:
        pass

    def fit(self, samples: List[dict], labels: List[int]) -> None:
        self.samples_, self.labels_ = samples, labels
        self._compute_possible_parametes()
        self.results_ = _multiprocess(self.run, self.n_runs_)
        self.histories = [result["history"] for result in self.results_]
        self.best_solutions_ = [result["best_solution"] for result in self.results_]

    def _compute_possible_parametes(self) -> None:
        i = 0
        self.possible_parameters_ = []
        samples_positive = [sample for sample, label in zip(self.samples_, self.labels_) if label == 1]
        for sample in samples_positive:
            if sample not in self.possible_parameters_:
                sample["index"] = i
                self.possible_parameters_.append(sample)
                i += 1

    def _get_from_possible_parameters(self) -> dict:
        max_value = len(self.possible_parameters_) - 1
        index = random.randint(0, max_value)
        return deepcopy(self.possible_parameters_[index])

    def _create_solutions(self) -> List[tuple]:
        solutions = []
        for _ in range(self.n_solutions_):
            new_solution = [self._get_from_possible_parameters() for _ in range(self.n_parameters_)]
            new_fitness = self.get_fitness(self.samples_, self.labels_, new_solution)
            solutions.append((new_solution, new_fitness))
        return solutions

    @staticmethod
    def _select_best_solution(solutions: List[tuple]) -> None:
        sorted_solutions = sorted(solutions, key=lambda x: x[1].f1, reverse=True)
        return sorted_solutions[1][0]

    def evaluate(self, samples: List[dict], labels: List[int]) -> dict:
        f1, precision, recall = [], [], []
        for solution in self.best_solutions_:
            fitness = self.get_fitness(samples, labels, solution)
            f1.append(fitness.f1)
            precision.append(fitness.precision)
            recall.append(fitness.recall)
        return {
            "f1": f"{np.mean(f1):.4} ± {np.std(f1):.4}",
            "precision": f"{np.mean(precision):.4} ± {np.std(precision):.4}",
            "recall": f"{np.mean(recall):.4} ± {np.std(recall):.4}"
        }


def _multiprocess(func: Callable, n_runs: int) -> list:
    pool = Pool(processes=5)
    results = [pool.apply_async(func) for _ in range(n_runs)]
    pool.close()  # no more tasks will be submitted to the pool
    pool.join()  # wait for all tasks to finish before moving on
    return [result.get() for result in results]
