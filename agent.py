import torch
import random
import pygame
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer, ReplayBuffer, MultiLinearQNet
from plotme import TrainingPlot
from genetic import GeneticAlgorithm

"""
Deep Q-Learning 
Q new value = Q current   + Learn Rate * [ Reward + Discount Rate * Max Future Expected Reward - Q_current ]
Q_new(s,a) = Q_current(s,a) + ALPHA [R (s,a) + GAMMA MAX Q'(s',a') - Q_current(s,a) ]"""

#########################################################################################################################
"""
ALPHA - LEARNING RATE DQN

The learning rate determines the extent to which newly acquired information overrides old information. 
It regulates how much the Q-values are updated based on new experiences.
A higher learning rate means faster updates, but it might lead to instability or overshooting optimal values."""

#########################################################################################################################
"""
GAMMA - DISCOUNT RATE DQN

The discount factor signifies the importance of future rewards compared to immediate rewards. 
It determines how much the agent values future rewards over immediate ones. 
A higher discount factor values long-term rewards more, influencing the agent’s decision-making."""

#########################################################################################################################
"""EPSILON - EXPLORATION RATE Q_current -> Q_new using Bellman Equation

Loss = E [(rt + GAMMA * max(Q_st+1, a', theta-) - Q (st, at, theta))^2
Loss = greedy strategy ---> starts high and decreases over time

if random_number < epsilon:
   select_random_action()
else:
   select_action_with_highest_q_value()"""

# VARIABLES

MAX_MEMORY = 100_000 # Maximum memory for the agent  

param_ranges = {
    # Continuous parameters
    'learning_rate': (0.001, 0.1), # Alpha / Higher values allow faster learning, while lower values ensure more stability
    'discount_factor': (0.9, 0.999), #Gamme / Closer to 1 indicate future rewards are highly important, emphasizing long-term rewards
    'dropout_rate': (0.1, 0.5), # Higher drops out a more neurons -> prevent overfit in complex models or datasets with limited samples
    'exploration_rate': (0.1, 0.5), #Epsilon /Higher more exploration -> Possibly better actions /Lower -> More stability using learned policy
    
    # Discrete parameters
    # 'batch_size': [10, 100, 250, 500, 1000, 2000, 5000], # Number of experiences sampled from the replay buffer for training.
    # 'activation_function': ['relu', 'sigmoid', 'tanh'],
    # 'optimizer': ['adam', 'sgd', 'rmsprop'], 
    
    # Integer parameters (num_inputs, num_outputs of NN)
    # 'num_hidden_layers': [1, 2, 3, 4, 5],
    # 'neurons_per_layer': [32, 64, 128, 256, 512, 1024]

    # Other parameters
    #'MAX_MEMORY' -> capacity of replay memory
}

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
POPULATION_SIZE = 20
CHROMOSOME_LENGTH, NUM_GENERATIONS = 15, 5 



class QLearningAgent:

    def __init__(self, parameters):
        self.n_games = 0 # Number of games played
        self.epsilon = parameters.get('exploration_rate', 0.3) # Parameter for exploration-exploitation trade-off
        self.gamma = parameters.get('discount_factor', 0.9) # Discount factor for future rewards
        self.dropout_rate = parameters.get('dropout_rate', 0.2)
        self.lr = parameters.get('learning_rate', 0.001)
        #self.lr = alpha = 0.001
        #self.dropout_rate = 0.2
        #self.epsilon = 0.3
        #self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # Replay memory for storing experiences
        input_size, hidden_size, output_size, num_hidden_layers = 11, 256, 3, 1
        #num_hidden_layers = parameters.get('num_hidden_layers', 1)
        #activation_function = parameters.get('activation_function', 'relu')
        self.model = LinearQNet(input_size, hidden_size, output_size, self.dropout_rate, num_hidden_layers, activation_function = 'relu')
        
        #optimizer = parameters.get('optimizer','adam')
        self.trainer = QTrainer(self.model, lr = self.lr, gamma = self.gamma, optimizer_name = 'adam') 
        #self.batch_size = parameters.get('batch_size', batch_size)
        self.batch_size = 1000 # Learning rate for the model
        self.replay_buffer = ReplayBuffer(capacity = self.batch_size)
        self.target_model = LinearQNet(input_size, hidden_size, output_size, self.dropout_rate, num_hidden_layers, activation_function = 'relu')
        self.target_model.load_state_dict(self.model.state_dict())  # Sync initial weights


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

    """Store experience (state, action, reward, next_state, done) in memory // MAX_MEMORY"""
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    """Sample from memory and perform a training step using QTrainer"""
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, ReplayBuffer, self.batch_size)

    """Perform single training step using a single experience tuple (short-term memory)"""
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, ReplayBuffer, self.batch_size)

    """Select actions based on an epsilon-greedy strategy, balancing exploration and exploitation in the agent's decision-making process.
    - 1. Epsilon Decay; 2. Action Selection; 3. Outcome"""
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

def create_random_parameters(param_ranges): # For initialization of a population
    return {param: random.uniform(*ranges) if isinstance(ranges, tuple) else random.choice(ranges)
            for param, ranges in param_ranges.items()}

def train():
    plotter = TrainingPlot() # To store game scores for plotting
    plot_scores = [] # To store scores for plotting  
    plot_mean_scores = []  # To store mean scores for plotting 
    total_score = 0
    record = 0
    visited_positions = set() # Unique Values
    same_positions_counter = 0

    # Initialize the agent with random parameters from param_range
    agent = QLearningAgent(random_params) 
    game = SnakeGameAI() # Initialize the game environment
    genetic = GeneticAlgorithm(
                            POPULATION_SIZE = POPULATION_SIZE,
                            CHROMOSOME_LENGTH = CHROMOSOME_LENGTH,
                            param_ranges = param_ranges, 
                            MUTATION_RATE = MUTATION_RATE,
                            NUM_GENERATIONS = NUM_GENERATIONS,
                            game = game, 
                            neural_network_architecture = agent.model
                            )
    
    game_metrics_list = []  # List to store game metrics (score, record, steps, collisions, same positions)
    
    while True: 
        # Capture the state before taking an action
        state_old = agent.get_state(game) 
        # Determine the next move/action using the RL agent
        final_move = agent.get_action(state_old)
        # Execute the selected move and observe the game's response
        reward, done, score, collisions, steps = game.play_step(final_move)
        # Update the agent's internal score
        game.score = score
        # Capture the new state after the action
        state_new = agent.get_state(game)
        # Train the agent using this experience
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        # Get the current position of the snake's head
        current_position = (game.snake[0].x, game.snake[0].y)

        if current_position in visited_positions:
            same_positions_counter += 1

        # Add the current position to the set of visited positions
        visited_positions.add(current_position)
            
        # # Check if the snake's head position has been visited before
        # same_positions = len(visited_positions) != len(set(visited_positions))


        if done: # He is dead

            # Initialize the game for the next iteration
            game._init_game()
            agent.n_games += 1
            agent.train_long_memory()

            # Update the highest record if the current score surpasses
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update plot data for visualization
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plotter.update(plot_scores, plot_mean_scores)

            # Store game metrics in a dictionary
            game_metrics = {
                'score': score,
                'record': record,
                'steps': steps,
                'collisions': collisions,
                'same_positions': same_positions_counter
            }

            game_metrics_list.append(game_metrics)

            # Pass game metrics to the genetic algorithm after each game
            _, best_params, _ = genetic.genetic(NUM_GENERATIONS, score = score, record = record, steps = steps, 
                                                    collisions = collisions, same_positions_counter = same_positions_counter,  
                                                    game_metrics_list = game_metrics_list)
            
            keys = [
                    'learning_rate',
                    'discount_factor',
                    'dropout_rate',
                    'exploration_rate',
                ]
            
            # Pair keys with corresponding values from best_params
            best_parameters = {key: value for key, value in zip(keys, best_params)}
                            
            # Reinitialize the agent with the best parameters for the next game
            agent = QLearningAgent(parameters = best_parameters)
            same_positions_counter = 0
            steps = 0

def train_RL():
    plotter = TrainingPlot() # To store game scores for plotting
    plot_scores = [] # To store scores for plotting  
    plot_mean_scores = []  # To store mean scores for plotting 
    total_score = 0
    record = 0
    visited_positions = set() # Unique Values
    same_positions_counter = 0
    agent = QLearningAgent() 
    game = SnakeGameAI() # Initialize the game environment
    game_metrics_list = []  # List to store game metrics (score, record, steps, collisions, same positions)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, collisions, steps = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # Get the current position of the snake's head
        current_position = (game.snake[0].x, game.snake[0].y)

        if current_position in visited_positions:
            same_positions_counter += 1

        # Add the current position to the set of visited positions
        visited_positions.add(current_position)

        if done:
            # train long memory, plot result
            game._init_game()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update plot data for visualization
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plotter.update(plot_scores, plot_mean_scores)

            # Store game metrics in a dictionary
            game_metrics = {
                'score': score,
                'record': record,
                'steps': steps,
                'collisions': collisions,
                'same_positions': same_positions_counter
            }

            game_metrics_list.append(game_metrics)



if __name__ == "__main__": 
    random_params = create_random_parameters(param_ranges)
    train()
    # train_RL()