import matplotlib.pyplot as plt
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


class BaseMethods:

    @staticmethod
    def multiprocess(func: Callable, n_runs: int) -> list:
        pool = Pool(processes=10)
        results = [pool.apply_async(func) for _ in range(n_runs)]
        pool.close()  # no more tasks will be submitted to the pool
        pool.join()  # wait for all tasks to finish before moving on
        return [result.get() for result in results]

    def objective_function(
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
    def plot_training_curve(histories) -> None:
        _, axis = plt.subplots(nrows=3, ncols=1, figsize=(16, 8))

        scores = {"f1": [], "precision": [], "recall": []}
        for history in histories:
            new_scores = {"f1": [], "precision": [], "recall": []}
            for generation in history:
                best_harmony = max([
                    (harmony[1].f1, harmony[1].precision, harmony[1].recall)
                    for harmony in generation
                ])
                new_scores["f1"].append(best_harmony[0])
                new_scores["precision"].append(best_harmony[1])
                new_scores["recall"].append(best_harmony[2])
            scores["f1"].append(new_scores["f1"])
            scores["precision"].append(new_scores["precision"])
            scores["recall"].append(new_scores["recall"])

        for ax, (key, list_of_lists) in zip(axis, scores.items()):
            for values in list_of_lists:
                ax.plot(values)
            ax.grid(True)
            ax.set_ylabel(key, size=14, labelpad=10)

        plt.show()

    def evaluate(self, samples: List[dict], labels: List[int], solution: List[dict]) -> dict:
        fitness = self.objective_function(samples, labels, solution)
        return {
            "f1": fitness.f1,
            "precision": fitness.precision,
            "recall": fitness.recall,
        }

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
