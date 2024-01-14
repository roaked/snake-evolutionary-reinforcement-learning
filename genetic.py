import random
import numpy as np

##############################################################################################################################
"""
To do list:
- IT IS MUTATING BETWEEN PARAMETERES!!!!!!!!!!!!!!!!!!!!!! discount_factor = 'adams optimizer' is certainly wrong

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

POPULATION_SIZE = 5

"""Typical empirical values range from 20 to 50 individuals in a generation"""
 
CHROMOSOME_LENGTH, NUM_GENERATIONS = 15, 3 # param_ranges items

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
    
    
    def fitness_function(self, score, record, steps, collisions, same_positions_counter):
        # Initialize fitness to 0
        fitness = 0

        # Calculate the score fitness
        score_fitness = score**3 * 20000
        fitness += score_fitness

        # Calculate the record fitness
        record_fitness = record**3 * 5000
        fitness += record_fitness

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
        mutated_individual = individual.copy()

        if isinstance(individual, list):
            # Handle list mutation
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    try:
                        mutated_individual[i] = max(min(individual[i] + random.uniform(-0.1, 0.1), 1.0), 0.0)
                    except ValueError:
                        print("Conversion to float failed. 'gene' might not be a numeric value.")
        elif isinstance(individual, dict):
            # Handle dictionary mutation
            for param, gene in individual.items():
                if random.random() < mutation_rate:
                    try:
                        mutated_gene = max(min(gene + random.uniform(-0.1, 0.1), 1.0), 0.0)
                        mutated_individual[param] = mutated_gene
                    except ValueError:
                        print(f"Mutation failed for parameter '{param}'.")

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

            for parameters in self.population:
                fitness = self.fitness_function(score, record, steps, collisions, same_positions_counter)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_parameters = parameters.copy()

                best_agents.append((parameters, fitness))

            print(f"Generation {generation}: Best Parameters - {best_parameters}, Fitness - {best_fitness}")

        return best_agents, best_parameters, best_fitness


#######################################################################################################################################
