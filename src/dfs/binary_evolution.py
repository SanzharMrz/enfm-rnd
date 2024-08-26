import numpy as np

class BinaryEvolution():
    """
    A class that implements the evolutionary algorithm for binary arrya

    Args:
        n (int): The number of bits in the binary array
        population_size (int): The number of individuals in the population
        crossover_prob (float): The probability of crossover
        mutation_prob (float): The probability of mutation
    """

    def __init__(self, n: int, population_size: int, crossover_prob=0.5, mutation_prob=0.5):
        self.n = n
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = np.random.randint(low=0, high=2, size=(self.population_size, self.n))
        self.prev_metrics = [np.inf for i in range(self.population_size)]

    def get_new_population(self):
        """
        Generate a new population of individuals

        Returns:
            new_population (np.array): A new population of individuals
        """
        new_population = np.zeros((self.population_size, self.n))
        for i in range(self.population_size):
            child = np.copy(self.population[i])
            crossover = np.random.rand(1)[0]
            if crossover < self.crossover_prob:
                parent2 = np.random.choice([j for j in range(self.population_size) if j != i], 1)
                crossover_point = np.random.randint(1, self.n)
                child[crossover_point:] = self.population[parent2, crossover_point:]
    
            mutation = np.random.rand(1)[0]
            if mutation < self.mutation_prob:
                mutation_point = np.random.randint(0, self.n)
                child[mutation_point] = 1 - child[mutation_point]
            new_population[i] = child
        return new_population

    def update_population(self, new_population, metrics):
        """"
        Update the population with the new population based on fitness values

        Args:
            new_population (np.array): A new population of individuals
            metrics (list): A list of metrics for each individual in the population
        """
        for i in range(self.population_size):
            if metrics[i] < self.prev_metrics[i]:
                self.population[i] = new_population[i]
                self.prev_metrics[i] = metrics[i]