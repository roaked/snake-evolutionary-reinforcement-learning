import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Dueling_QNet(nn.Modele):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()



# class Linear_QNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x

#     def save(self, file_name='model.pth'):
#         model_folder_path = './model'
#         if not os.path.exists(model_folder_path):
#             os.makedirs(model_folder_path)

#         file_name = os.path.join(model_folder_path, file_name)
#         torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, target_update_freq = 1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        ## double Q learn
        self.target_model = Linear_QNet(input_size, hidden_size, output_size)  # Target Network
        self.target_update_freq = target_update_freq
        self.update_count = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        
        with torch.no_grad():
                    best_action = torch.argmax(self.model(next_state), dim=1, keepdim=True)
                    target_Q = self.target_model(next_state)
                    Q_new = reward + (1 - done) * self.gamma * target_Q.gather(1, best_action)

        for idx in range(len(done)):
            target[idx][action[idx]] = Q_new[idx].squeeze().item()

        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

        #     target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())