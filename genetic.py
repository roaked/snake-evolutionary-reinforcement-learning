import random, time, itertools
import numpy as np

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


    def __init__(self, population_size, chromosome_length, param_ranges):
        self.population_size = population_size #20 to 50 individuals empirical value
        self.param_ranges = param_ranges #dictionary upstairs
        self.chromosome_length = chromosome_length # c_length = n_in + neurons*n_out + n_out = (11*256) + 256 + (256*3) + 3 = 3075
        self.population = self.generate_population()



    def generate_population(self, population_size, param_ranges): #Random init or heuristic init (using prior info)

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
    
    
    def fitness_function(self, score, record, steps, collisions, same_positions): #from current state
        """Need to implement input for same_positions, score, record, steps"""

        """Metrics"""
        # record -> highest score achieved thus far -> (total_score variable in agent.py)
        # score -> current score (# apples eaten)
        # collisions -> number of deaths thus far -> should be = (game number - 1)
        # steps -> average number of steps to eat food
        

        """Weights"""
        weight_score = 0.6
        weight_collisions, MAX_COLLISIONS = 0.2, 200 # Max collisions
        weight_steps, MAX_POSSIBLE_STEPS = 0.2, 200 # Avg steps to eat food

         
        """Normalize metrics"""
        normalized_score = score / record if record != 0 else 0
        normalized_collisions = 1 - (collisions / MAX_COLLISIONS) #Penalizes alot of deaths/collisions
        normalized_steps = 1 - (steps / MAX_POSSIBLE_STEPS)  #Penalizes excessive steps to incentivize efficiency

        # Penalize collisions (20%)
        penalty_collisions = 0.2 if normalized_collisions > 0.5 else 0  # Penalize frequent collisions

        # Penalty for 100 steps without score increase (10%)
        penalty_steps = 0.10 if normalized_steps < 0.5 else 0  

        # Penalty for revisiting same positions (15%)
        penalty_same_positions = 0.15 if same_positions > 0 else 0 

        # Efficiency decay (5%)
        efficiency_decay = max(0, (steps - score) / MAX_POSSIBLE_STEPS)  # Measure deviation from optimal steps for score
        penalty_efficiency_decay = 0.05 * efficiency_decay  # Penalize for efficiency decay

        fitness = (
            (normalized_score ** weight_score) *
            (normalized_steps ** weight_steps) *
            (normalized_collisions ** weight_collisions) -
            penalty_steps - penalty_collisions  -
            penalty_same_positions - penalty_efficiency_decay
        )

        return fitness
    
    def calculate_population_fitness(self, population): #pop size usually between 20 and 50 for a generation
        fitness_scores = []
        for individual in population:

            score = individual['score'] 
            record = individual['record']
            steps = individual['steps'] 
            collisions = individual['collisions']  
            same_positions = individual['same_positions'] 

            # Calculate fitness using the function you provided
            fitness = self.fitness_function(score, record, steps, collisions, same_positions)

            # Store the fitness score for the current individual
            fitness_scores.append(fitness)

        return fitness_scores
    
    def selection(self, population, fitness_scores):
        # Normalize fitness scores to probabilities
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores] # List Comprehension - Probabilities Array

        # Select based on fitness (roulette wheel selection) // replace = True means one individual can be picked more than 1 time
        selected_indices = np.random.choice(len(population), size = self.population_size, replace = True, p = probabilities)

        # Create a new population based on the selected indices
        new_population = [population[idx] for idx in selected_indices] # List Comprehension - New population Array

        return new_population
    
    ##############################################################################################################################
    """# Not using but could select the 'Elite' individuals with highest p - num_elites is the length of selecting individuals"""
    def elitist_selection(population, fitness_scores, num_elites): 
        # Get indices of individuals sorted by fitness (descending order)
        sorted_indices = sorted(range(len(fitness_scores)), key = lambda i: fitness_scores[i], reverse=True)
        
        # Select the top individuals as elites
        new_population = [population[idx] for idx in sorted_indices[:num_elites]]
    
        return new_population # Diff. size / length
    ##############################################################################################################################


    selected_population = selection(population = population, fitness_scores = fitness_scores) # Put at the end of code after implementation
    parent1, parent2 = random.sample(selected_population, 2) # Put at the end of code after implementation
    
    """Single-point crossover for two parent individuals. Can explore two-point crossover, uniform crossover, elitist crossover, etc."""
    def crossover(self, parent1, parent2):

        assert len(parent1) == len(parent2) # Only if same len

        # Crossover point // Similar to Bin Packing Problem (BPP)
        crossover_point = random.randint(1, len(parent1) - 1)

        # Create offspring by combining parent genes
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        return offspring1, offspring2
    
    
    offspring1, offspring2 = crossover(parent1, parent2)  # Put at the end of code after implementation

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

ga = GeneticAlgorithm(population_size = 20, chromosome_length = 3075, param_ranges = param_ranges) 

# pop size usually between 20 and 50
# chromosome length = (11 * 256) + 256 + (256 * 3) + 3 = 3075 (neural network)

