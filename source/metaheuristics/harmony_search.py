import random
from copy import deepcopy
from typing import List
from .utils import BaseMetaheuristic


class HarmonySearch(BaseMetaheuristic):

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
        BaseMetaheuristic.__init__(
            self,
            n_parameters,
            n_harmonies,
            n_generations,
            n_runs
        )

    def run(self) -> dict:
        self.history_: List[list] = []
        harmony_memory = self._create_solutions()
        for i in range(self.n_generations_ * self.n_solutions_):
            new_harmony = self._improvise(harmony_memory)
            new_fitness = self.get_fitness(self.samples_, self.labels_, new_harmony)
            self._update_harmony_memory(harmony_memory, new_harmony, new_fitness)
            self._update_history(harmony_memory, i)
        best_harmony = self._select_best_solution(harmony_memory)
        return {"history": self.history_, "best_solution": best_harmony}

    def _improvise(self, harmony_memory: List[tuple]) -> List[dict]:
        harmony: List[dict] = [{}] * self.n_parameters_
        for i in range(self.n_parameters_):
            if random.random() < self.hmcr_:
                harmony[i] = self._get_from_harmony_memory(harmony_memory)[i]
                self._tweak_parameter(harmony[i], self.par_)
            else:
                harmony[i] = self._get_from_possible_parameters()
        return harmony

    def _get_from_harmony_memory(self, harmony_memory: List[tuple]) -> dict:
        index = random.randint(0, self.n_solutions_-1)
        return deepcopy(harmony_memory[index][0])

    @staticmethod
    def _update_harmony_memory(
        solutions: List[tuple],
        new_solution: List[dict],
        new_fitness: float
    ) -> None:
        if (new_solution, new_fitness) not in solutions:
            worst_index = 0
            worst_fitness = float("inf")
            for i, (_, fitness) in enumerate(solutions):
                if fitness < worst_fitness:
                    worst_index = i
                    worst_fitness = fitness
            if new_fitness > worst_fitness:
                solutions[worst_index] = (new_solution, new_fitness)

    def _update_history(self, harmony_memory: List[tuple], iteration: int) -> None:
        if iteration % self.n_solutions_ == 0:
            self.history_.append([harmony[1] for harmony in harmony_memory])
