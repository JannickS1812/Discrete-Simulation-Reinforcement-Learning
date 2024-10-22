from .q_learning_agent import QLearningAgent
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
import torch
from copy import deepcopy


class ExperienceReplay(Dataset):
    def __init__(self, model, target_model=None, max_memory=100, alpha=0.5, gamma=0.99, transform=None, target_transform=None):
        self.model = model
        self.target_model = model if target_model is None else target_model
        self.memory = []
        self.max_memory = max_memory
        self.alpha = alpha
        self.gamma = gamma
        self.transform = transform
        self.target_transform = target_transform

    def get_not_seen_action(self, state, filter):
        valid_actions = [i for i in range(6) if filter[i] == True]
        for a in valid_actions:
            if not any([all(l[0][0] == state) and l[0][1] == a for l in self.memory]):
                return a
        return -1

    def remember(self, experience, game_over):
        # Save a state to memory

        if not any([all(l[0][0] == experience[0]) and l[0][1] == experience[1] and l[0][2] == experience[2] and all(l[0][3] == experience[3]) for l in self.memory]):
            self.memory.append([experience, game_over])
            #if len(self) % 10 == 0:
            #    print("ReplayBuffer Length", len(self))
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_weights(self):
        states = np.array([row[0][0] for row in self.memory])
        actions = [row[0][1] for row in self.memory]
        rewards = [row[0][2] for row in self.memory]
        states_new = np.array([row[0][3] for row in self.memory])
        goal_state = [row[1] for row in self.memory]

        q_states = self.model[states]
        q_states_new = self.target_model[states_new]

        td = [rewards[i] + self.gamma * max(q_states_new[i]) - q_states[i, actions[i]] if goal_state[i] else rewards[i] - q_states[i, actions[i]] for i in range(len(self))]

        temporal_diffs = (np.abs(td) + 1e-7)**self.alpha

        return temporal_diffs / np.sum(temporal_diffs)

    def update_model(self, model):
        self.model = model

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        s, a, r, s_new = self.memory[idx][0]
        goal_state = self.memory[idx][1]
        features = np.array(s)
        # init labels with old prediction (and later overwrite action a)
        label = self.model[s]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma * max(self.model[s_new])

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
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions), )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters())
        self.loss_fn = loss_fn
        self.transform = transform

    def __getitem__(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().to(self.device)
        prediction = self(state)
        return prediction.cpu().detach().numpy()

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

    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))


class DeepDuelingQTable(DeepQTable):
    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepDuelingQTable, self).__init__(number_of_states, number_of_actions, Optimizer, loss_fn, transform)
        self.input_network = nn.Sequential(
            nn.Linear(number_of_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())
        self.value_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.advantage_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions))
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters())

    def forward(self, x):
        features = self.input_network(x)
        values = self.value_network(features)
        advantages = self.advantage_network(features)
        return values + (advantages - advantages.mean())


class DeepQLearningAgent(QLearningAgent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=10, R_Max=100,
                 q_table_file="deep_q_table.pth", batch_size=50, Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss(),
                 ModelClass=DeepQTable, prioritized_replay=True, trainings_per_step = 1):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration,
                         R_Max=R_Max, q_table_file=q_table_file)
        all_states = np.array(self.states)
        min_values = np.amin(all_states, axis=0)
        max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
        transform = lambda x: (x - min_values) / (max_values - min_values)
        if q_table is None:
            self.q_table = self.create_model(Optimizer, loss_fn, transform, ModelClass)
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, transform=transform, max_memory=100000)
        self.loss_history = []
        self.prioritized_replay = prioritized_replay
        self.trainings_per_step = trainings_per_step

    def create_model(self, Optimizer, loss_fn, transform, ModelClass):
        return ModelClass(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)

        if self.prioritized_replay:
            weights = self.experience_replay.get_weights()

        if len(self.experience_replay) > self.batch_size:
            for i in range(self.trainings_per_step):
                if self.prioritized_replay:
                    sampler = WeightedRandomSampler(weights, len(weights))
                    train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, sampler=sampler)
                else:
                    train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
                self.loss_history += self.q_table.perform_training(train_loader)

    def save(self, appendix = ''):
        self.q_table.save_model(self.file + str(appendix))

    def load(self):
        self.q_table.load_model(self.file)


class DoubleDeepQLearningAgent(DeepQLearningAgent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100,
                 q_table_file="double_deep_q_table.pth", batch_size=10, Optimizer=torch.optim.Adam,
                 loss_fn=nn.MSELoss(), ModelClass=DeepQTable, update_interval=20):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration,
                         R_Max=R_Max, batch_size=batch_size, q_table_file=q_table_file, Optimizer=Optimizer,
                         loss_fn=loss_fn, ModelClass=ModelClass)
        self.online_q_table = deepcopy(self.q_table)
        self.update_count = 0
        self.update_interval = update_interval

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)
        if len(self.experience_replay) > self.batch_size:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
            # training is now done on the online network
            self.loss_history += self.online_q_table.perform_training(train_loader)
            self.update_count += 1
            # update target network with online network
            if self.update_count % self.update_interval == 0:
                self.q_table = deepcopy(self.online_q_table)
                self.experience_replay.update_model(self.q_table)


class DeepQLearningAgentPlantSim(DeepQLearningAgent):

    def act(self):
        # perception
        s = self.problem.get_current_state()

        if s.any():
            filter = self.problem.filter_valid_actions(s)
            q_values = [q if filter[i] else -np.inf for i, q in enumerate(self.q_table[s])]
            a = self.actions[np.argmax(q_values)]
            return a
        else:
            return None

    def eval(self, max_steps=500):
        self.problem.reset()
        self.problem.start()
        self.problem.unpause_simulation()

        steps = 0
        a = None
        is_goal_state = False
        cumsum = 0
        while steps < max_steps and not self.problem.is_goal_state(None) and not self.problem.do_break_episode():
            if self.problem.simulation_needs_action():
                s_new = self.problem.get_current_state()
                r = self.problem.get_reward(s_new)
                cumsum += r
                if s_new.any():
                    self.problem.pause_simulation()

                    filter = self.problem.filter_valid_actions(s_new)
                    q_values = [q if filter[i] else -np.inf for i, q in enumerate(self.q_table[s_new]) ]
                    a = self.actions[np.argmax(q_values)]

                    # act
                    self.problem.act(a)
                    s = s_new

                    steps += 1
                    self.problem.unpause_simulation()

        return cumsum/steps, self.problem.evaluation, self.problem.get_time()

    def train(self, max_steps=500, random_action=0):
        self.problem.reset()
        self.problem.start()
        self.problem.unpause_simulation()

        steps = 0
        a = None
        is_goal_state = False
        cumsum = 0
        while steps < max_steps and not self.problem.is_goal_state(None) and not self.problem.do_break_episode():
            if self.problem.simulation_needs_action():
                s_new = self.problem.get_current_state()
                r = self.problem.get_reward(s_new)
                cumsum += r
                if s_new.any():
                    self.problem.pause_simulation()
                    if tuple(s_new) not in self.N_sa.keys():
                        self.N_sa[tuple(s_new)] = np.zeros(len(self.actions))
                        self.q_table[s_new] = np.zeros(len(self.actions))

                    if a is not None:
                        self.N_sa[tuple(s)][a] += 1
                        is_goal_state = self.problem.is_goal_state(s_new)
                        self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(is_goal_state))

                    if is_goal_state:
                        return self.q_table, self.N_sa

                    '''Epsilon Greedy with do one action at least once'''
                    #random_action = 0 if random_action < 0.05 else random_action
                    '''Do action if not in replay buffer'''
                    a = self.experience_replay.get_not_seen_action(s_new, filter=self.problem.filter_valid_actions(s_new))
                    if a == -1:
                        if np.random.random() < random_action:
                            p = np.array(self.problem.filter_valid_actions(s_new))
                            a = np.random.choice(self.actions, p=p / p.sum())
                        else:
                            filter = self.problem.filter_valid_actions(s_new)
                            q_values = [q if filter[i] else -np.inf for i, q in enumerate(self.q_table[s_new]) ]
                            a = self.actions[np.argmax(q_values)]

                    # act
                    self.problem.act(a)
                    s = s_new

                    steps += 1
                    self.problem.unpause_simulation()
        return cumsum/steps, self.problem.evaluation


    def choose_GLIE_action(self, q_values, N_s, filter=None):
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = self.R_Max / 2 + q_values
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)

        if filter is not None:
            max_values *= np.array(filter)

        # assure that we do not dived by zero
        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values) / max_values.size
        else:
            probabilities = max_values / max_values.sum()

        if filter is not None:
            probabilities *= np.array(filter)
            probabilities /= probabilities.sum()

        # select action according to the (q) values
        if np.random.random() < (self.max_N_exploration + 0.00001) / (np.max(N_s) + 0.00001):

            print("Explore ", (self.max_N_exploration + 0.00001) / (np.max(N_s) + 0.00001), probabilities)
            try:
                action = np.random.choice(self.actions, p=probabilities)
            except:
                print('Error')
        else:
            print("Exploit ", (self.max_N_exploration + 0.00001) / (np.max(N_s) + 0.00001), probabilities)
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes)
            action = self.actions[action_index]
        return action


