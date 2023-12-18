import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)  # Example: Adding dropout with p=0.2

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)  # Applying dropout
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, target_update_freq = 1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # Zero the gradients, compute loss, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()