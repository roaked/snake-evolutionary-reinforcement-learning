import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class LinearQNet(nn.Module): #
    def __init__(self, input_size, hidden_size, output_size, dropout_value, num_hidden_layers, activation_function):
        super().__init__()

        self.layers = nn.ModuleList() # ModuleList to store dynamically created layers
        
        self.layers.append(nn.Linear(input_size, hidden_size)) # Input
        self.layers.append(nn.Dropout(dropout_value))  
        self.layers.append(self.get_activation(activation_function)) 

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))  # Hidden layer
            self.layers.append(nn.Dropout(dropout_value))  
            self.layers.append(self.get_activation(activation_function))  

        self.layers.append(nn.Linear(hidden_size, output_size)) # Output

        for layer in self.layers: # Init with Xavier weights
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Activation function '{name}' not supported.")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, optimizer_name):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer = self.get_optimizer(optimizer_name)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.target_update_counter = 0


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done, ReplayBuffer, batch_size):

        # if len(ReplayBuffer) < batch_size:  ###ERROR?
        #     return
        
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        pred = self.model(state)
        target = pred.clone()

        # Q-learning update rule
        # Handling single-dimensional state and action tensors
        if state.dim() == 1:  # (1,x)
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicting Q-values based on current state-action pair
        pred = self.model(state)

        # Clone the prediction for updating
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) #target model maybe?

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # Zero the gradients, compute loss, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.update_target()

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        self.train_step(state, action, reward, next_state, done)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())