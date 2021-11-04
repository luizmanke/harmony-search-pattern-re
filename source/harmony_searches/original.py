import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from typing import List
from .base import ANY_VALUE, BaseMethods


class OriginalHS(BaseMethods):

    def __init__(
        self,
        par: float,
        hmcr: float,
        n_parameters: int,
        n_harmonies: int,
        n_generations: int,
        n_runs: int
    ) -> None:
        self.par_ = par
        self.hmcr_ = hmcr
        self.n_parameters_ = n_parameters
        self.n_harmonies_ = n_harmonies
        self.n_generations_ = n_generations
        self.n_runs_ = n_runs

    def fit(self, samples: List[dict], labels: List[int]) -> None:
        self.samples_, self.labels_ = samples, labels
        self._compute_possible_parametes()
        self.histories_ = [self._run() for _ in tqdm(range(self.n_runs_))]
        self._select_best_harmonies()

    def plot_training_curve(self) -> None:
        BaseMethods.plot_training_curve(self.histories_)

    def evaluate(self, samples: List[dict], labels: List[int]) -> dict:
        results = {"f1": [], "precision": [], "recall": []}
        for harmony in self.best_harmonies_:
            result = BaseMethods.evaluate(self, samples, labels, harmony)
            results["f1"].append(result["f1"])
            results["precision"].append(result["precision"])
            results["recall"].append(result["recall"])
        results["f1"] = f"{np.mean(results['f1']):.4} ± {np.std(results['f1']):.4}"
        results["precision"] = f"{np.mean(results['precision']):.4} ± {np.std(results['precision']):.4}"
        results["recall"] = f"{np.mean(results['recall']):.4} ± {np.std(results['recall']):.4}"
        return results

    def _run(self) -> List[list]:
        history = []
        harmony_memory = self._create_harmony_memory()
        for i in range(self.n_generations_ * self.n_harmonies_):
            new_harmony = self._improvise(harmony_memory, i // self.n_harmonies_)
            new_fitness = self.objective_function(self.samples_, self.labels_, new_harmony)
            self._update_harmony_memory(harmony_memory, new_harmony, new_fitness)
            if i % self.n_harmonies_ == 0:
                history.append(deepcopy(harmony_memory))
        return history

    def _compute_possible_parametes(self) -> None:
        i = 0
        self.possible_parameters_ = []
        samples_positive = [
            sample for sample, label in zip(self.samples_, self.labels_) if label == 1
        ]
        for sample in samples_positive:
            if sample not in self.possible_parameters_:
                sample["index"] = i
                self.possible_parameters_.append(sample)
                i += 1

    def _create_harmony_memory(self) -> List[dict]:
        harmony_memory = []
        for _ in range(self.n_harmonies_):
            new_harmony = [self._get_from_possible_parameters() for _ in range(self.n_parameters_)]
            new_fitness = self.objective_function(self.samples_, self.labels_, new_harmony)
            harmony_memory.append((new_harmony, new_fitness))
        return harmony_memory

    def _improvise(self, harmony_memory: List[dict], generation: int) -> dict:
        harmony = [{}] * self.n_parameters_
        for i in range(self.n_parameters_):
            if random.random() < self._get_hmcr(generation):
                harmony[i] = self._get_from_harmony_memory(harmony_memory)[i]
                if random.random() < self._get_par(generation):
                    self._pitch_adjustment(harmony[i])
            else:
                harmony[i] = self._get_from_possible_parameters()
        return harmony

    def _get_from_harmony_memory(self, harmony_memory: List[dict]) -> dict:
        index = random.randint(0, self.n_harmonies_-1)
        return deepcopy(harmony_memory[index][0])

    def _get_hmcr(self, generation: int) -> float:
        return self.hmcr_

    def _get_par(self, generation: int) -> float:
        return self.par_

    def _pitch_adjustment(self, parameter: dict) -> None:

        random_value = random.random()
        original_parameter = self.possible_parameters_[parameter["index"]]

        has_changed = False
        for key, values in parameter.items():

            if key == "index":
                continue

            indexes = np.arange(0, len(values))
            np.random.shuffle(indexes)
            for index in indexes:
                baseline = ANY_VALUE if random_value < 0.5 else original_parameter[key][index]
                if values[index] != baseline:
                    values[index] = baseline
                    has_changed = True
                    break

            if has_changed:
                break

    def _get_from_possible_parameters(self) -> dict:
        max_value = len(self.possible_parameters_) - 1
        index = random.randint(0, max_value)
        return deepcopy(self.possible_parameters_[index])

    def _update_harmony_memory(
        self,
        harmony_memory: List[dict],
        new_harmony: dict,
        new_fitness: float
    ) -> None:
        if (new_harmony, new_fitness) not in harmony_memory:
            worst_index = None
            worst_fitness = float("inf")
            for i, (_, fitness) in enumerate(harmony_memory):
                if fitness < worst_fitness:
                    worst_index = i
                    worst_fitness = fitness
            if new_fitness > worst_fitness:
                harmony_memory[worst_index] = (new_harmony, new_fitness)

    def _select_best_harmonies(self) -> None:
        self.best_harmonies_ = []
        for history in self.histories_:
            best_harmony = None
            best_fitness = float("-inf")
            for harmony in history[-1]:
                fitness = harmony[1].f1
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_harmony = harmony[0]
            self.best_harmonies_.append(best_harmony)
