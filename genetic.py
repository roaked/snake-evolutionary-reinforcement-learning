import random
from model import QTrainer
from agent import QLearningAgent


####Implementation based on Bin Packing Genetic Algorithm 

###IMMUTABLE VARIABLES

GENERATIONS = 1000
LEARNING_RATE_GA = (0.1, 0.9)

class GeneticAlgorithm:
    def __init__(self, population_size, fitness, chromosome_length):
        self.population_size = population_size
        self.fit = fitness
        self.param_ranges = param_ranges
        self.chromosome_length = chromosome_length
        self.population = self.generate_population()

    def generate_population(self):
        # Generate initial population of parameters
        population = []
        for _ in range(self.population_size):
            params = {param: random.uniform(param_range[0], param_range[1])
                      for param, param_range in self.param_ranges.items()}
            population.append(params)
        return population
    
    #def initialize_population(self):
    #   return [[random.uniform(0, 1) for _ in range(self.chromosome_length)] for _ in range(self.population_size)]
    
    def fitness_function(self, record, deaths, avg_steps, penalties):
        # record -> highest score achieved -> (total_score variable in agent.py)
        # death -> number of deaths
        # avg steps -> average number of steps to eat food
        # penalties (flexible) -> number of times it did 200 steps without eating any food
        fitness = record * 500 - deaths * 150 - avg_steps * 100 - penalties * 100
        
    
    def selection(self, agent): #based on fitness function
        if fit == bestFit:
            return population[individual]
        pass
    
    def crossover(self, parent1, parent2): #offspring from 2 parents
        pass

    def mutation(self, chromosome): #randomize changes
        pass
    
    def evolve(self, generations):
        best_agents = [] # store best
        self.population = self.initialize_population()

        for generation in range(generations):
            # Evaluate fitness for each chromosome in the population
            fitness_scores = [self.fitness_function(chromosome) for chromosome in self.population]

            # Select high-performing chromosomes (using tournament selection)
            selected_chromosomes = self.selection(fitness_scores)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_chromosomes[i]
                parent2 = selected_chromosomes[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])

            # Replace the least fit part of the population with offspring
            elite_count = int(self.population_size * 0.1)  # Keep top 10% as elite
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]

            for idx in elite_indices:
                offspring[idx] = self.population[idx]  # Preserve elite chromosomes

            # Fill the rest with offspring (biased towards better fitness)
            self.population = random.sample(offspring, self.population_size - elite_count)

            # Store information on the best agent of this generation
            best_chromosome = max(self.population, key=self.fitness_function)
            best_fitness = self.fitness_function(best_chromosome)
            best_agents.append((best_chromosome, best_fitness))

            print(f"Generation {generation}: Best Chromosome - {best_chromosome}, Fitness - {best_fitness}")

        return best_agents

ga = GeneticAlgorithm(population_size=10, fitness=2, chromosome_length=10) # pop size = 10

