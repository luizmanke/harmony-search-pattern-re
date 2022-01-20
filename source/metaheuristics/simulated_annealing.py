import math
import numpy as np
import random
from copy import deepcopy
from typing import List
from .utils import ANY_VALUE, BaseMetaheuristic


class SimulatedAnnealing(BaseMetaheuristic):

    def __init__(
        self,
        step_chance: float,
        initial_temperature: float,
        temperature_decay: float,
        n_parameters: int,
        n_individuals: int,
        n_generations: int,
        n_runs: int
    ) -> None:
        self.step_chance_ = step_chance
        self.initial_temperature_ = initial_temperature
        self.temperature_decay_ = temperature_decay
        BaseMetaheuristic.__init__(
            self,
            n_parameters,
            n_individuals,
            n_generations,
            n_runs
        )

    def run(self) -> dict:
        self.history_: List[list] = []
        current_population = self._create_solutions()
        best_population = deepcopy(current_population)
        for i in range(self.n_generations_):
            candidate_individuals = self._take_a_step(current_population)
            candidate_population = self._evaluate_individuals(candidate_individuals)
            best_population = self._update_best_population(best_population, candidate_population)
            current_population = self._update_current_population(current_population, candidate_population, i)
            self._update_history(best_population)
        best_individual = self._select_best_solution(best_population)
        return {"history": self.history_, "best_solution": best_individual}

    def _take_a_step(self, population: List[tuple]) -> List[list]:
        new_individuals = []
        for one_tuple in deepcopy(population):
            individual = one_tuple[0]
            for parameter in individual:

                original_parameter = self.possible_parameters_[parameter["index"]]
                if random.random() >= self.step_chance_:
                    continue

                has_changed = False
                for key, values in parameter.items():

                    if key == "index":
                        continue

                    indexes = np.arange(0, len(values))
                    random.shuffle(indexes)
                    for index in indexes:
                        baseline = ANY_VALUE if random.random() < 0.5 else original_parameter[key][index]
                        if values[index] != baseline:
                            values[index] = baseline
                            has_changed = True
                            break

                    if has_changed:
                        break

            new_individuals.append(individual)
        return new_individuals

    def _evaluate_individuals(self, individuals: List[list]) -> List[tuple]:
        population: List[tuple] = []
        for individual in individuals:
            fitness = self.get_fitness(self.samples_, self.labels_, individual)
            population.append((individual, fitness))
        return population

    @staticmethod
    def _update_best_population(
        best_population: List[tuple],
        candidate_population: List[tuple]
    ) -> List[tuple]:
        new_best_population: List[tuple] = []
        for best_tuple, candidate_tuple in zip(best_population, candidate_population):
            if best_tuple[1].f1 >= candidate_tuple[1].f1:
                new_best_population.append(best_tuple)
            else:
                new_best_population.append(candidate_tuple)
        return new_best_population

    def _update_current_population(
        self,
        current_population: List[tuple],
        candidate_population: List[tuple],
        iteration: int
    ) -> List[tuple]:
        new_current_population = []
        t = self.initial_temperature_ * self.temperature_decay_**iteration
        for current_tuple, candidate_tuple in zip(current_population, candidate_population):
            difference = candidate_tuple[1].f1 - current_tuple[1].f1
            metropolis = math.exp(-difference / t)
            if difference < 0 or random.random() < metropolis:
                new_current_population.append(candidate_tuple)
            else:
                new_current_population.append(current_tuple)
        return new_current_population

    def _update_history(self, population: List[tuple]):
        self.history_.append([individual[1] for individual in population])
