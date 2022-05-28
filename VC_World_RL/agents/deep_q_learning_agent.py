import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .q_learning_agent import QLearningAgent



class ExperienceReplay(Dataset):
    def __init__(self, model, target_model=None, max_memory=100, gamma=.99, alpha=0.5, transform=None, target_transform=None):
        self.model = model
        self.target_model = model if target_model is None else target_model
        self.memory = []
        self.max_memory = max_memory
        self.gamma = gamma
        self.alpha = alpha
        self.transform = transform
        self.target_transform = target_transform


    def remember(self, states, game_over):
        self.memory.append([states, game_over])

        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def update_model(self, model):
        self.model = model

    def get_weights(self):
        temporal_diffs = np.zeros((len(self),))
        for i in range(len(self)):
            s, a, r, s_new = self.memory[i][0]
            goal_state = self.memory[i][1]

            if goal_state:
                td = r - self.model[s][a]
            else:
                td = r + self.gamma * max(self.target_model[s_new]) - self.model[s][a]
            td = abs(td) + 1e-7  # small eps to ensure that each sample has a non-zero propability of being drawn
            temporal_diffs[i] = td**self.alpha

        return temporal_diffs / np.sum(temporal_diffs)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        s, a, r, s_new = self.memory[idx][0]
        goal_state = self.memory[idx][1]
        features = np.array(s)
        label = self.model[s]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma*max(self.target_model[s_new])

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        features = torch.from_numpy(features).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        return features, label


class DeepQTable(nn.Module):
    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, number_of_actions),
        )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)
        self.optimizer = Optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.transform = transform

    def __getitem__(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().to(self.device)
        return self.model(state).cpu().detach().numpy()

    def __setitem__(self, state, value):
        # ignoring setting to values
        pass

    def forward(self, x):
        return self.model(x)

    def perform_training(self, dataloader):
        loss_history = []

        (X, y) = next(iter(dataloader))
        # Compute prediction and loss
        pred = self(X)

        loss = self.loss_fn(pred, y)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_history.append(loss)
        return loss_history


class DeepQLearningAgent(QLearningAgent):
    def __init__(self, problem, file, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100, Optimizer=Adam, loss_fn=nn.MSELoss(), batch_size=20, prioritized_replay=True):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration, R_Max=R_Max, q_table_file=file)
        if q_table is None:
            all_states = np.array(self.states)
            min_values = np.amin(all_states, axis=0)
            max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
            transform = lambda x: (x - min_values) / (max_values - min_values)
            self.q_table = self.create_model(Optimizer, loss_fn, transform)

        self.prioritized_replay = prioritized_replay
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, transform=transform)
        self.loss_history = []

    def create_model(self, Optimizer, loss_fn, transform):
        return DeepQTable(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)

        if self.prioritized_replay:
            weights = self.experience_replay.get_weights()
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
        self.loss_history += self.q_table.perform_training(train_loader)

    def save_q_table(self):
        torch.save(self.q_table.state_dict(), self.file)

    def load_q_table(self):
        self.q_table.load_state_dict(torch.load(self.file))


class DoubleDeepQLearningAgent(QLearningAgent):
    def __init__(self, problem, file, q_table=None, tau=0.1, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100, Optimizer=Adam, loss_fn=nn.MSELoss(), batch_size=20, prioritized_replay=True):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration, R_Max=R_Max, q_table_file=file)
        if q_table is None:
            all_states = np.array(self.states)
            min_values = np.amin(all_states, axis=0)
            max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
            transform = lambda x: (x - min_values) / (max_values - min_values)
            self.q_table = self.create_model(Optimizer, loss_fn, transform)

        self.prioritized_replay = prioritized_replay
        self.target_q_table = deepcopy((self.q_table))
        self.tau = tau
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, target_model=self.target_q_table, transform=transform)
        self.loss_history = []


    def create_model(self, Optimizer, loss_fn, transform):
        return DeepQTable(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)

        if self.prioritized_replay:
            weights = self.experience_replay.get_weights()
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
        self.loss_history += self.q_table.perform_training(train_loader)

        # polyak averaging
        for target_param, param in zip(self.target_q_table.model.parameters(), self.q_table.model.parameters()):
            target_param.data = self.tau * param.data + (1-self.tau) * target_param.data


    def save_q_table(self):
        torch.save(self.q_table.state_dict(), self.file)

    def load_q_table(self):
        self.q_table.load_state_dict(torch.load(self.file))

