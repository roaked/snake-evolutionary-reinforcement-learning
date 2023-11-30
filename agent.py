##creating venv with miniconda
##conda pipenv -n "name here" python=3.7
##conda activate "name here"


import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

###### IMMUTABLE VARIABLES

MAX_MEMORY = 100_000 # Maximum memory for the agent  
BATCH_SIZE = 1000 # Batch size for training
ALPHA = 0.001  # Learning rate for the model
LEARNING_RATE_GA = (0.1, 0.9)


# Deep Q-Learning 
# Q new value = Q current   + Learn Rate * [ Reward + Discount Rate * Max Future Expected Reward - Q_current ]
# Q_new(s,a) = Q_current(s,a) + ALPHA [R (s,a) + GAMMA MAX Q'(s',a') - Q_current(s,a) ]

##########################################################################################################################
#ALPHA - LEARNING RATE DQN

#The learning rate determines the extent to which newly acquired information overrides old information. 
#It regulates how much the Q-values are updated based on new experiences.
#A higher learning rate means faster updates, but it might lead to instability or overshooting optimal values.

##########################################################################################################################
#GAMMA - DISCOUNT RATE DQN

#The discount factor signifies the importance of future rewards compared to immediate rewards. 
#It determines how much the agent values future rewards over immediate ones. 
#A higher discount factor values long-term rewards more, influencing the agent’s decision-making.

##########################################################################################################################
#EPSILON - EXPLORATION RATE Q_current -> Q_new using Bellman Equation

#Loss = E [(rt + GAMMA * max(Q_st+1, a', theta-) - Q (st, at, theta))^2
#Loss = greedy strategy ---> starts high and decreases over time

#if random_number < epsilon:
#    select_random_action()
#else:
#    select_action_with_highest_q_value()

# Genetic Algorithm - Fitness Function

class QLearningAgent:

    def __init__(self):
        self.n_games = 0 # Number of games played
        self.epsilon = 0 # Parameter for exploration-exploitation trade-off
        self.gamma = 0.9 # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY) # Replay memory for storing experiences
        self.model = Linear_QNet(11, 256, 3) # Neural network model
        self.trainer = QTrainer(self.model, lr=ALPHA, gamma=self.gamma) # QTrainer for model training


    def get_state(self, game):
        # Function to obtain the state representation based on the game state
        # Generates information about dangers, direction, and food location
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store experience (state, action, reward, next_state, done) in memory
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        # Sample from memory and perform a training step using QTrainer
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Perform a single training step using a single experience tuple
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Select actions based on an epsilon-greedy strategy
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = [] # To store game scores for plotting
    plot_mean_scores = [] # To store mean scores for plotting   
    total_score = 0 # Total score across all games
    record = 0 # Record score obtained in a game
    agent = QLearningAgent() # Initialize the agent
    game = SnakeGameAI() # Initialize the game environment

    while True:
        # Get the old state from the game
        state_old = agent.get_state(game)

        # Choose an action based on the old state
        final_move = agent.get_action(state_old)

        # Perform the chosen action in the game and get the new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train the agent using short memory (single experience)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience in the agent's memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train the agent's long-term memory and plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update scores for plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores) # Plot the game scores and mean scores


#class GeneticAlgorithm:
    def __init__(self, population_size, param_ranges):
        self.population_size = population_size
        self.param_ranges = param_ranges

    def generate_population(self):
        # Generate initial population of parameters
        population = []
        for _ in range(self.population_size):
            params = {param: random.uniform(param_range[0], param_range[1])
                      for param, param_range in self.param_ranges.items()}
            population.append(params)
        return population
    
    def evolve(self):
        population = self.generate_population()

        for generation in range(num_generations):
            # Train Q-learning agents with parameters from the population
            agents = [QLearningAgent(params) for params in population]
            for agent in agents:
                agent.train()
                agent.evaluate()  # Evaluate the agent's performance

            # Select top-performing agents for genetic operations
            top_agents = select_top_agents(agents)

            # Apply genetic operations (crossover and mutation)
            new_population = crossover_and_mutation(top_agents)

            population = new_population

        # Extract the best-performing agent from the final population
        best_agent_params = get_best_agent_params(population)
        best_agent = QLearningAgent(best_agent_params)
        return best_agent


if __name__ == '__main__':
    train()

# if __name2__ == '__main__':

#     # Define parameter ranges for genetic algorithm
#     param_ranges = {
#         "learning_rate": LEARNING_RATE_GA,
#         "epsilon": (0.1, 0.5),
#         # Add other Q-learning hyperparameters here
#     }

#     # Initialize genetic algorithm -> Optimize agent parameters (Defining which parameters)
#     ga = GeneticAlgorithm(population_size=10, param_ranges=param_ranges)

#     # Evolve and find the best Q-learning agent parameters
#     best_agent = ga.evolve()

#     # Train the best Q-learning agent with the optimal parameters
#     best_agent.train()

#     train()