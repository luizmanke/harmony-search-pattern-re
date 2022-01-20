import numpy as np
import random
from copy import deepcopy
from typing import List
from .utils import ANY_VALUE, BaseMetaheuristic


class GeneticAlgorithm(BaseMetaheuristic):

    def __init__(
        self,
        n_elites: int,
        mutation_rate: float,
        n_parameters: int,
        n_individuals: int,
        n_generations: int,
        n_runs: int
    ) -> None:
        self.n_elites_ = n_elites
        self.mutation_rate_ = mutation_rate
        BaseMetaheuristic.__init__(
            self,
            n_parameters,
            n_individuals,
            n_generations,
            n_runs
        )

    def run(self) -> dict:
        self.history_: List[list] = []
        population = self._create_solutions()
        for i in range(self.n_generations_):
            new_individuals = self._crossover(population)
            new_individuals = self._mutate(new_individuals)
            population = self._update_population(new_individuals)
            self._update_history(population)
        best_individual = self._select_best_solution(population)
        return {"history": self.history_, "best_solution": best_individual}

    def _crossover(self, population: List[tuple]) -> List[list]:
        new_individuals = []

        sorted_population = deepcopy(sorted(population, key=lambda x: x[1].f1, reverse=True))
        new_individuals.extend([x[0] for x in sorted_population[:self.n_elites_]])

        for _ in range(self.n_elites_, len(population)):
            parent_1 = parent_2 = self._roulette_wheel(population)
            while parent_1 == parent_2:
                parent_2 = self._roulette_wheel(population)
            crossover_point = random.randint(0, len(parent_1))
            new_individual = parent_1[:crossover_point] + parent_2[crossover_point:]
            new_individuals.append(new_individual)

        return new_individuals

    @staticmethod
    def _roulette_wheel(population: List[tuple]) -> List[dict]:
        total = sum([individual[1].f1 for individual in population])
        probabilities = [individual[1].f1/total for individual in population]
        individual = random.choices(population, weights=probabilities)[0]
        return individual[0]

    def _mutate(self, individuals: List[list]) -> List[list]:
        new_individuals = []
        for i, individual in enumerate(deepcopy(individuals)):
            if i >= self.n_elites_:
                for parameter in individual:

                    original_parameter = self.possible_parameters_[parameter["index"]]
                    if random.random() >= self.mutation_rate_:
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

    def _update_population(self, individuals: List[list]) -> List[tuple]:
        population: List[tuple] = []
        for individual in individuals:
            fitness = self.get_fitness(self.samples_, self.labels_, individual)
            population.append((individual, fitness))
        return population

    def _update_history(self, population: List[tuple]):
        self.history_.append([individual[1] for individual in population])
