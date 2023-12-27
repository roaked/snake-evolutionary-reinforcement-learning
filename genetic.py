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
    'discount_factor': (0.9, 0.999), #Gamma / Closer to 1 indicate future rewards are highly important, emphasizing long-term rewards
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


    def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, param_ranges, MUTATION_RATE, NUM_GENERATIONS, game, neural_network_architecture):
        self.population_size = POPULATION_SIZE 
        self.num_generations = NUM_GENERATIONS
        self.param_ranges = param_ranges #dictionary upstairs
        self.chromosome_length = CHROMOSOME_LENGTH # c_length empri
        self.population = self.generate_population(self.population_size, self.param_ranges, self.chromosome_length)
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE
        self.game = game
        self.neural_network_architecture = neural_network_architecture

    def get_chromosome(self):
        return {
            'score': self.score,
            'record': self.record,
            'steps': self.steps,
            'collisions': self.collisions,
            'same_positions': self.same_positions
        }
    

    def generate_population(self, population_size, param_ranges, chromosome_length): #Random init or heuristic init (using prior info)
        population = []
        chromosome_length = len(param_ranges)
        for _ in range(population_size):
            params = {}
            for param, value_range in param_ranges.items():
                if isinstance(value_range, tuple):  # Random init for continuous parameters
                    params[param] = random.uniform(value_range[0], value_range[1])
                elif isinstance(value_range, list):  # Discrete parameters
                    params[param] = random.choice(value_range)
                elif isinstance(value_range, int):  # Integer parameters
                    params[param] = random.randint(0, value_range)
                elif isinstance(value_range, str):  # String parameters
                    params[param] = value_range  # Set the string value directly
            population.append(params)

        return population
    
    
    def fitness_function(self, score, record, steps, collisions, same_positions_counter): #from current state
        """Metrics and weights"""
        weight_score = 0.75
        weight_steps, MAX_POSSIBLE_STEPS = 0.25, 200

        """Normalize metrics"""
        normalized_score = score / record if record != 0 else 0
        normalized_steps = 1 - (steps / MAX_POSSIBLE_STEPS) if MAX_POSSIBLE_STEPS != 0 else 0

        # Penalty for revisiting same positions > 30
        penalty_same_positions = 0.05 if same_positions_counter > 30 else 0

        # Efficiency decay (5%)
        efficiency_decay = max(0, (steps - score) / MAX_POSSIBLE_STEPS)
        penalty_efficiency_decay = 0.05 * efficiency_decay

        # Calculate fitness
        fitness = (
            (normalized_score * weight_score) +
            (normalized_steps * weight_steps) -
            penalty_same_positions - penalty_efficiency_decay
        )

        return max(0, fitness)  # Ensure non-negative fitness
    
    def calculate_population_fitness(self, population, game_metrics_list):

        """We should add fitness scores for all the population...."""

        fitness_scores = []

        if len(game_metrics_list) >= 5:
            last_5_game_metrics = game_metrics_list[-5:] # Ensure there are at least 5 game metrics
        else:
            last_5_game_metrics = game_metrics_list # Less than 5 games, consider all available

        for individual_metrics in last_5_game_metrics:
            # Extract individual game metrics from the dictionary
            score = individual_metrics['score']
            record = individual_metrics['record']
            steps = individual_metrics['steps']
            collisions = individual_metrics['collisions']
            same_positions_counter = individual_metrics['same_positions']

            # Calculate fitness based on game performance
            fitness = self.fitness_function(score, record, steps, collisions, same_positions_counter)
            fitness_scores.append(fitness)

        return fitness_scores
    
    ##############################################################################################################################
    
    #fitness_scores = calculate_population_fitness(population)
    
    ##############################################################################################################################
    
    def selection(self, population, fitness_scores):
        # Normalize fitness scores to probabilities
        print('\n')
        print("Fitness Scores:", fitness_scores)
        print('\n')

        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            probabilities = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [fitness / total_fitness for fitness in fitness_scores]

        # Ensure probabilities array size matches population size
        while len(probabilities) < len(population):
            probabilities.append(0.0)

        # Select based on fitness (roulette wheel selection) // replace = True means one chromosome can be picked more than 1 time
        selected_indices = np.random.choice(
            len(population), 
            size=self.population_size, 
            replace=True, 
            p=probabilities / np.sum(probabilities)  # Normalize probabilities to sum up to 1
        )

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


    # selected_population = selection(population = population, fitness_scores = fitness_scores) # Put at the end of code after implementation
    # parent1, parent2 = random.sample(selected_population, 2) # Put at the end of code after implementation
    
    """Single-point crossover for two parent individuals. Can explore two-point crossover, uniform crossover, elitist crossover, etc."""
    def crossover(self, parent1, parent2, crossover_rate):

        if isinstance(parent1, dict) and isinstance(parent2, dict):
            # Convert dictionary values to lists
            parent1 = list(parent1.values())
            parent2 = list(parent2.values())

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
    
    
    # offspring1, offspring2 = crossover(parent1, parent2, crossover_rate = CROSSOVER_RATE)  # Put at the end of code after implementation


    """According to Genetic Algorithm, after crossover (breeding), we apply mutation to the resulting offspring to introduce
    small changes to their genetic material depending on the mutation rate, this helps explores new areas of solution space"""
    def mutation(self, individual, mutation_rate):

        mutated_individual = []
        for gene in individual:
            if random.random() < mutation_rate:
                try: # Assuming 'gene' is a string that needs to be converted to a numerical type
                    gene = int(gene)  # Convert 'gene' to an integer
                    mutated_gene = 1 - gene
                except ValueError:
                    print("Conversion to integer failed. 'gene' might not be a numeric value.")
            else:
                mutated_gene = gene
            mutated_individual.append(mutated_gene)
        return mutated_individual
    
    # mutated_offspring1 = mutation(offspring1, mutation_rate = MUTATION_RATE)
    # mutated_offspring2 = mutation(offspring2, mutation_rate = MUTATION_RATE)



    ##############################################################################################################################
    """My notes

    population = generate_population(population_size = POPULATION_SIZE, param_ranges = param_ranges, chromosome_length = CHROMOSOME_LENGTH)
    fitness_scores = calculate_population_fitness(population = population)
    selected_population = selection(population = population, fitness_scores = fitness_scores) # Put at the end of code after implementation
    parent1, parent2 = random.sample(selected_population, 2) # Put at the end of code after implementation
    offspring1, offspring2 = crossover(parent1, parent2)  # Put at the end of code after implementation
    mutated_offspring1 = mutation(offspring1, mutation_rate = MUTATION_RATE)
    mutated_offspring2 = mutation(offspring2, mutation_rate = MUTATION_RATE)"""

     ##############################################################################################################################

    def genetic(self, num_generations, score, record, steps, collisions, same_positions_counter, game_metrics_list):

        best_parameters = None 
        best_fitness = float('-inf')  
        best_agents = [] # Store best

        for generation in range(num_generations):
            # Evaluate fitness for each chromosome in the population
            #fitness_scores = [self.fitness_function(chromosome) for chromosome in self.population]
            fitness_scores = self.calculate_population_fitness(self.population, game_metrics_list) 
            """Compare both"""

            # Select high-performing chromosomes (using tournament selection)
            selected_population = self.selection(self.population, fitness_scores)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected_population)-1, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1] # Consecutive pairs
                child1, child2 = self.crossover(parent1, parent2, CROSSOVER_RATE)
                child1 = self.mutation(child1, self.mutation_rate)
                child2 = self.mutation(child2, self.mutation_rate)
                offspring.extend([child1, child2])



            # Replace the least fit part of the population with offspring
            elite_count = int(self.population_size * 0.1)  # Keep top 10% as elite
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        
            for idx in elite_indices:
                offspring[idx] = self.population[idx]  # Preserve elite chromosomes

            for item in offspring:
                if isinstance(item, dict):
                    print("Dictionary:")
                    for key, value in item.items():
                        print(f"{key}: {value}")
                elif isinstance(item, list):
                    print("List:")
                    for value in item:
                        print(value)
                else:
                    print("Unknown type")
                print()

            # Fill the rest with offspring (biased towards better fitness)
            self.population = random.sample(offspring, self.population_size - elite_count)

            # Store information on the best agent of this generation

            for game_metrics in game_metrics_list:
                score = game_metrics['score']
                record = game_metrics['record']
                steps = game_metrics['steps']
                collisions = game_metrics['collisions']
                same_positions = game_metrics['same_positions']
                
                # Use these values as needed within your code
                # For instance, print them
                print(f"Score: {score}, Record: {record}, Steps: {steps}, Collisions: {collisions}, Same Positions: {same_positions}")

            current_best_chromosome = max(self.population, key=lambda game_metrics_list: self.fitness_function
                                          (game_metrics_list['score'], game_metrics_list['record'], game_metrics_list['steps'],
                                            game_metrics_list['collisions'], game_metrics_list['same_positions_counter']))
            current_best_fitness = self.fitness_function(current_best_chromosome)
            best_agents.append((current_best_chromosome, current_best_fitness))

            if current_best_fitness > best_fitness:
                best_parameters = current_best_chromosome  # Update best parameters
                best_fitness = current_best_fitness
            print(f"Generation {generation}: Best Chromosome - {current_best_chromosome}, Fitness - {current_best_fitness}")

        return best_agents, best_parameters, best_fitness


# pop size usually between 20 and 50
# chromosome length = (11 * 256) + 256 + (256 * 3) + 3 = 3075 (neural network)

# if __name__ == "__main__": 
#     ga = GeneticAlgorithm(population_size = POPULATION_SIZE, chromosome_length = CHROMOSOME_LENGTH, param_ranges = param_ranges,
#                           mutation_rate = MUTATION_RATE) #Initialized  
#     best_agents = GeneticAlgorithm().genetic(num_generations = NUM_GENERATIONS)
#     print(best_agents)
    

#######################################################################################################################################
    

class GeneticOldFunctions():
        
        def calculate_population_fitness(self, population, score, record, steps, collisions, same_positions_counter): 
            fitness_scores = []  #pop size usually between 20 and 50 for a generation

            default_values = {'score': 0, 'record': 0, 'steps': 0, 'collisions': 0, 'same_positions': 0}

            # Initializing the population list / Randomize
            population = [default_values.copy() for _ in range(POPULATION_SIZE)]

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
