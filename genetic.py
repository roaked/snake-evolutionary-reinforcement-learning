import random, time, itertools
import numpy as np

##############################################################################################################################
"""
To do list:
- Fitness function optimization -- Retrieve all input variables from other functions
- Implement GeneticAlgorithm() object in the agent.py function
- Check 'param_ranges' dictionary / hash map for new optimization params of the Deep Q Network
- Results and test
"""
##############################################################################################################################

"""What we want or can optimize in the Deep-Q-Network using Genetic Algorithm"""

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

"""Variables to use for the Genetic Algorithm"""

MUTATION_RATE = 0.1

"""(...) values typically range between 1% to 10%. A mutation rate of 1% implies that, on average, 
1 in every 100 genes will undergo a mutation per generation. Higher mutation rates, like 5% or 10%, 
can introduce more exploration but might risk losing some beneficial traits learned through crossover."""

CROSSOVER_RATE = 0.8

"""Typical values for the crossover rate in genetic algorithms often range between 0.6 and 0.9. 
(...)higher crossover rate, closer to 1.0, favors more exploration (diversity), 
(...)lower crossover rate, closer to 0.6, emphasizes exploitation (convergence)"""

POPULATION_SIZE = 20

"""Typical empirical values range from 20 to 50 individuals in a generation"""
 
CHROMOSOME_LENGTH, NUM_GENERATIONS = 15, 5 # param_ranges items

class GeneticAlgorithm:


    def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, param_ranges, MUTATION_RATE, NUM_GENERATIONS):
        self.population_size = POPULATION_SIZE 
        self.num_generations = NUM_GENERATIONS
        self.param_ranges = param_ranges #dictionary upstairs
        self.chromosome_length = CHROMOSOME_LENGTH # c_length empri
        self.population = self.generate_population(self.population_size, self.param_ranges, self.chromosome_length)
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE

    def generate_population(self, population_size, param_ranges, chromosome_length): #Random init or heuristic init (using prior info)
        #check if working

        population = []
        for _ in range(population_size):
            params = {}
            for _ in range(chromosome_length):
                for param, value_range in param_ranges.items():

                    # #Heuristic Initialization
                    # if param == 'learning_rate':
                    #     params[param] = 0.01  # Heuristic init for learning rate --> Check agent.py  
                    # elif param == 'dropout_rate':
                    #     params[param] = 0.2  # Heuristic init for dropout rate --> Check game.py    
                    # elif param == 'activation_function':
                    #     params[param] = 'relu'  # Heuristic init for activation function --> Default

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
        for chromosome in population:

            score = chromosome['score'] 
            record = chromosome['record']
            steps = chromosome['steps'] 
            collisions = chromosome['collisions']  
            same_positions = chromosome['same_positions'] 

            # Calculate fitness using the function you provided
            fitness = self.fitness_function(score, record, steps, collisions, same_positions)

            # Store the fitness score for the current chromosome
            fitness_scores.append(fitness)

        return fitness_scores
    
    ##############################################################################################################################
    
    fitness_scores = calculate_population_fitness(population)
    
    ##############################################################################################################################
    
    def selection(self, population, fitness_scores):
        # Normalize fitness scores to probabilities
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores] # List Comprehension - Probabilities Array

        # Select based on fitness (roulette wheel selection) // replace = True means one chromosome can be picked more than 1 time
        selected_indices = np.random.choice(len(population), size = self.population_size, replace = True, p = probabilities)

        # Create a new population based on the selected indices
        new_population = [population[idx] for idx in selected_indices] # List Comprehension - New population Array

        return new_population
    
    ##############################################################################################################################
    """# Not using but could select the 'Elite' individuals with highest p - num_elites is the length of selecting individuals"""
    def elitist_selection(population, fitness_scores, num_elites): 
        # Get indices of individuals sorted by fitness (descending order)
        elite_sorted_indices = sorted(range(len(fitness_scores)), key = lambda i: fitness_scores[i], reverse=True)
        
        # Select the top individuals as elites
        new_population = [population[idx] for idx in elite_sorted_indices[:num_elites]]
    
        return new_population, elite_sorted_indices # Diff. size / length
    ##############################################################################################################################


    selected_population = selection(population = population, fitness_scores = fitness_scores) # Put at the end of code after implementation
    parent1, parent2 = random.sample(selected_population, 2) # Put at the end of code after implementation
    
    """Single-point crossover for two parent individuals. Can explore two-point crossover, uniform crossover, elitist crossover, etc."""
    def crossover(self, parent1, parent2, crossover_rate):

        assert len(parent1) == len(parent2) # Only if same len

        if random.random() < crossover_rate:
            # Crossover point
            crossover_point = random.randint(1, len(parent1) - 1)

            # Create offspring by combining parent genes
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

            return offspring1, offspring2
        else:
            return parent1, parent2 # If crossover doesn't happen, return the parents
    
    
    offspring1, offspring2 = crossover(parent1, parent2, crossover_rate = CROSSOVER_RATE)  # Put at the end of code after implementation


    """According to Genetic Algorithm, after crossover (breeding), we apply mutation to the resulting offspring to introduce
    small changes to their genetic material depending on the mutation rate, this helps explores new areas of solution space"""
    def mutation(individual, mutation_rate):

        mutated_individual = []
        for gene in individual:
            if random.random() < mutation_rate:
                # Flip the bit
                mutated_gene = 1 - gene
            else:
                mutated_gene = gene
            mutated_individual.append(mutated_gene)
        return mutated_individual
    
    mutated_offspring1 = mutation(offspring1, mutation_rate = MUTATION_RATE)
    mutated_offspring2 = mutation(offspring2, mutation_rate = MUTATION_RATE)



    ##############################################################################################################################
    """My notes"""

    population = generate_population(population_size = POPULATION_SIZE, param_ranges = param_ranges, chromosome_length = CHROMOSOME_LENGTH)
    fitness_scores = calculate_population_fitness(population = population)
    selected_population = selection(population = population, fitness_scores = fitness_scores) # Put at the end of code after implementation
    parent1, parent2 = random.sample(selected_population, 2) # Put at the end of code after implementation
    offspring1, offspring2 = crossover(parent1, parent2)  # Put at the end of code after implementation
    mutated_offspring1 = mutation(offspring1, mutation_rate = MUTATION_RATE)
    mutated_offspring2 = mutation(offspring2, mutation_rate = MUTATION_RATE)

     ##############################################################################################################################

    def genetic(self, num_generations):

        best_agents = [] # Store best
        #population = self.generate_population(self.population_size, self.param_ranges) #No need, already initialized in __init__

        for generation in range(num_generations):
            # Evaluate fitness for each chromosome in the population
            fitness_scores = [self.fitness_function(chromosome) for chromosome in self.population]
            fitness_scores2 = self.calculate_population_fitness(self.population) 
            """Compare both"""

            # Select high-performing chromosomes (using tournament selection)
            selected_population = self.selection(self.population, fitness_scores)
            selected_population2 = self.selection(self.population, fitness_scores2)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, self.population_size, 2):

                parent1, parent2 = random.sample(selected_population, 2) # Randomized
                parent1, parent2 = selected_population[i]. selected_population[i+1] # Consecutive pairs
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, self.mutation_rate)
                child2 = self.mutation(child2, self.mutation_rate)
                offspring.extend([child1, child2])

            # Replace the least fit part of the population with offspring
            elite_count = int(self.population_size * 0.1)  # Keep top 10% as elite
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]

            """elitist_population could be useful"""
            elitist_population = self.elitist_selection(self.population, fitness_scores, num_elites = self.population_size * 0.1) 
        
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


# pop size usually between 20 and 50
# chromosome length = (11 * 256) + 256 + (256 * 3) + 3 = 3075 (neural network)

if __name__ == "__main__": 
    ga = GeneticAlgorithm(population_size = POPULATION_SIZE, chromosome_length = CHROMOSOME_LENGTH, param_ranges = param_ranges,
                          mutation_rate = MUTATION_RATE) #Initialized  
    best_agents = GeneticAlgorithm().genetic(num_generations = NUM_GENERATIONS)
    print(best_agents)
