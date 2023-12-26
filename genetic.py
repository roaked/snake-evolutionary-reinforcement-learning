import random, time, itertools
from model import QTrainer
from agent import QLearningAgent

"""Variables to optimize in the Deep-Q-Network using Genetic Algorithm"""

param_ranges = {
    # Continuous parameters
    'learning_rate': (0.001, 0.1), # Alpha / Higher values allow faster learning, while lower values ensure more stability
    'discount_factor': (0.9, 0.999), #Gamme / Closer to 1 indicate future rewards are highly important, emphasizing long-term rewards
    'dropout_rate': (0.1, 0.5), # Higher drops out a more neurons -> prevent overfit in complex models or datasets with limited samples
    'exploration_rate': (0.1, 0.5), #Epsilon /Higher more exploration -> Possibly better actions /Lower -> More stability using learned policy
    
    # Discrete parameters
    'batch_size': [10, 100, 250, 500, 1000, 2000, 5000], # Number of experiences sampled from the replay buffer for training.
    'activation_function': ['relu', 'sigmoid', 'tanh'],
    'optimizer': ['adam', 'sgd', 'rmsprop'], 
    
    # Integer parameters (num_inputs, num_outputs of NN)
    'num_hidden_layers': [1, 2, 3, 4, 5],
    'neurons_per_layer': [32, 64, 128, 256, 512, 1024]

    # Other parameters
    #'MAX_MEMORY' -> capacity of replay memory
}

class GeneticAlgorithm:

    
    def __init__(self, population_size, fitness, chromosome_length, param_ranges):
        self.population_size = population_size #20 to 50 individuals empirical value
        self.fit = fitness
        self.param_ranges = param_ranges #dictionary upstairs
        self.chromosome_length = chromosome_length # c_length = n_in + neurons*n_out + n_out = (11*256) + 256 + (256*3) + 3 = 3075
        self.population = self.generate_population()



    def generate_population(self, population_size, param_ranges): #random init or heuristic init (using prior info)
        population = []
        for _ in range(population_size):
            params = {}
            for param, value_range in param_ranges.items():

                #Heuristic Initialization
                if param == 'learning_rate':
                    params[param] = 0.01  # Heuristic init for learning rate
                    
                elif param == 'dropout_rate':
                    params[param] = 0.3  # Heuristic init for dropout rate
                    
                elif param == 'activation_function':
                    params[param] = 'relu'  # Heuristic init for activation function

                #Random Initialization
                if isinstance(value_range, tuple):  # Random init for continuous parameters
                    params[param] = random.uniform(value_range[0], value_range[1])
                elif isinstance(value_range, list):  # Discrete parameters
                    params[param] = random.choice(value_range)
                elif isinstance(value_range, int):  # Integer parameters
                    params[param] = random.randint(value_range[0], value_range[1])
                elif isinstance(value_range, str):  # String parameters
                    params[param] = value_range  # Set the string value directly
            population.append(params)
        return population
    
    
    def fitness_function(self, record, deaths, avg_steps, penalties):
        # record -> highest score achieved -> (total_score variable in agent.py)
        # death -> number of deaths
        # avg steps -> average number of steps to eat food
        # penalties (flexible) -> number of times it did 200 steps without eating any food
        fitness = record * 500 - deaths * 150 - avg_steps * 100 - penalties * 100
        
    
    def selection(self, agent): #based on fitness function
        bestFit = None
        #if fit == bestFit:
            #return population[individual]
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

ga = GeneticAlgorithm(population_size = 20, fitness = 2, chromosome_length = 3075) # pop size usually between 20 and 50

# chromosome length = (11 * 256) + 256 + (256 * 3) + 3 = 3075 (neural network)

