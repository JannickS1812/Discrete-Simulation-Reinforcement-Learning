from .agent import Agent
import torch
import numpy as np
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, Optimizer=torch.optim.Adam, learning_rate=3e-4,
                 gamma=0.99, transform=None, filter=None):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.backbone = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),)
        self.head = nn.Softmax(1)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.transform = transform
        self.filter = filter

    def forward(self, state):
        inp = self.transform(np.array(state))
        inp = torch.tensor(inp).float().unsqueeze(0).to(self.device)
        last_layer_activations = self.backbone(inp)
        if self.filter is not None:
            last_layer_activations += torch.log(torch.tensor(self.filter(state)).float().unsqueeze(0).to(self.device))
        return self.head(last_layer_activations)

    def get_max_action(self, state):
        #state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.argmax(np.squeeze(probs.cpu().detach().numpy()))
        return highest_prob_action

    def get_action(self, state):
        #state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, log_probabilities, rewards):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probabilities, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))


class MultiLabelPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, Optimizer=torch.optim.Adam, learning_rate=3e-4,
                 gamma=0.99, transform=None):
        super(MultiLabelPolicyNetwork, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            )
        self.heads = []
        for n in num_actions:
            self.heads.append(nn.Sequential(
                nn.Linear(64, n),
                nn.Softmax(1), )
            )
        self.num_actions = num_actions
        self.num_action_types = len(num_actions)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.transform = transform

    def forward(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        x = self.backbone(state)
        return torch.concat([head(x) for head in self.heads])

    def get_max_action(self, state):
        probs = np.squeeze(self.forward(state).cpu().detach().numpy())

        highest_prob_actions = []
        last_idx = 0
        for n in self.num_actions:
            highest_prob_actions.append(np.argmax(probs[last_idx:n]))
            last_idx = n

        return highest_prob_actions

    def get_action(self, state):
        probs = np.squeeze(self.forward(state).cpu().detach().numpy())
        highest_prob_actions = []
        log_probs = []
        last_idx = 0
        for n in self.num_actions:
            highest_prob_actions.append(np.random.choice(n, probs[last_idx:n]))
            log_probs.append(torch.log(probs.squeeze(0)[highest_prob_actions[-1]]))
            last_idx = n

        return highest_prob_actions, torch.concat(log_probs)

    def update_policy(self, log_probabilities, rewards):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probabilities, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))



class ReinforceAgent(Agent):

    def __init__(self, problem, Optimizer=torch.optim.Adam, gamma=0.99, file="policy.npy", PolicyClass=PolicyNetwork, filter=None):
        self.problem = problem
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        self.file = file
        all_states = np.array(self.states)
        min_values = np.amin(all_states, axis=0)
        max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
        self.transform = lambda x: (x - min_values) / (max_values - min_values)
        self.policy = PolicyClass(len(self.states[0]), len(self.actions), Optimizer=Optimizer, gamma=gamma,
                                  transform=self.transform, filter=filter)

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        action_index = self.policy.get_max_action(s)
        action = self.actions[action_index]
        return action

    def train(self):
        log_probabilities = []
        rewards = []
        step_count = 0
        while True:
            step_count += 1
            current_state = self.problem.get_current_state()
            rewards.append(self.problem.get_reward(current_state))
            s = current_state.to_state()
            action_index, log_prob = self.policy.get_action(s)
            log_probabilities.append(log_prob)
            if self.problem.is_goal_state(current_state):
                if len(rewards) > 1:
                    self.policy.update_policy(log_probabilities, rewards)
                print(f"Step Count: {step_count}")
                return
            # act
            action = self.actions[action_index]
            self.problem.act(action)

    def save(self):
        self.policy.save_model(self.file)

    def load(self):
        self.policy.load_model(self.file)


class ReinforceAgentPlantSim(Agent):


    def train(self, max_steps=500):
        self.problem.reset()
        self.problem.start()
        self.problem.unpause_simulation()

        log_probabilities = []
        rewards = []
        steps = 0
        s = None
        while steps < max_steps and not self.problem.is_goal_state(s):
            if self.problem.simulation_needs_action():
                s = self.problem.get_current_state()
                if s.any():
                    self.problem.pause_simulation()

                    steps += 1
                    a, log_prob = self.policy.get_action(s)
                    if s != 1:
                        log_probabilities.append(log_prob)
                        rewards.append(self.problem.get_reward(s))
                    self.problem.act(a)

                    self.problem.unpause_simulation()

        if len(rewards) > 1:
            self.policy.update_policy(log_probabilities, rewards)
            print(f"Step Count: {steps}")



