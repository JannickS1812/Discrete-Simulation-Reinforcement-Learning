import numpy as np

import torch

from .reinforce_agent import PolicyNetwork, MultiLabelPolicyNetwork, ReinforceAgent
from .deep_q_learning_agent import DeepQTable, DeepQLearningAgent


class Actor(PolicyNetwork):

    def update_policy(self, log_probability, q_value):
        policy_gradient = -log_probability * torch.tensor(q_value)

        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()


class MultiLabelActor(MultiLabelPolicyNetwork):

    def update_policy(self, log_probability, q_value):
        policy_gradient = -log_probability * torch.tensor(q_value)

        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()


class QActorCriticAgent(ReinforceAgent, DeepQLearningAgent):
    def __init__(self, problem, Optimizer=torch.optim.Adam,
                 gamma=0.99, actor_file="q_actor.npy", PolicyClass=Actor,
                 batch_size=10, ValueNetworkClass=DeepQTable,
                 critic_file="q_critic.npy"):
        DeepQLearningAgent.__init__(self, problem=problem,
                                    batch_size=batch_size, ModelClass=ValueNetworkClass,
                                    q_table_file=critic_file)
        ReinforceAgent.__init__(self, problem=problem,
                                Optimizer=Optimizer, gamma=gamma, file=actor_file,
                                PolicyClass=PolicyClass)

    def act(self):
        return ReinforceAgent.act(self)

    def train(self):
        current_state = self.problem.get_current_state()
        if self.problem.is_goal_state(current_state):
            return
        s = current_state.to_state()
        a, log_prob = self.policy.get_action(s)
        action = self.actions[a]
        self.problem.act(action)

        while True:
            current_state = self.problem.get_current_state()
            s_new = current_state.to_state()
            r = self.problem.get_reward(current_state)

            # get q-values
            q_value_s = self.q_table[s][a]

            # update policy
            self.policy.update_policy(log_prob, q_value_s)

            # q_table update
            is_goal_state = self.problem.is_goal_state(current_state)
            self.update_q_values(s, a, r, s_new, is_goal_state)
            if is_goal_state:
                return

            # get new action
            a, log_prob = self.policy.get_action(s)
            s = s_new
            # act
            action = self.actions[a]
            self.problem.act(action)


class RPB:
    def __init__(self):
        self.rpb = {}
    def push(self, s, a, r):
        s = str((s.tolist(), a))
        if s in self.rpb.keys():
            if self.rpb[s] != r:
                print("Error REWARD")
        else:
            self.rpb[s] = r


class QActorCriticAgentPlantSim(QActorCriticAgent):
    def __init__(self, problem, Optimizer=torch.optim.Adam,
                 gamma=0.99, actor_file="q_actor.npy", PolicyClass=Actor,
                 batch_size=10, ValueNetworkClass=DeepQTable,
                 critic_file="q_critic.npy"):
        DeepQLearningAgent.__init__(self, problem=problem,
                                    batch_size=batch_size, ModelClass=ValueNetworkClass,
                                    q_table_file=critic_file)
        ReinforceAgent.__init__(self, problem=problem,
                                Optimizer=Optimizer, gamma=gamma, file=actor_file,
                                PolicyClass=PolicyClass, filter=problem.filter_valid_actions)
        self.rpb = RPB()
        
    def act(self):
        return ReinforceAgent.act(self)

    def train(self, max_steps=500):

        self.problem.reset()
        self.problem.start()
        self.problem.unpause_simulation()
        steps = 0
        s_old = None

        cum_reward = 0
        while steps < max_steps and not self.problem.is_goal_state(s_old):#and not self.problem.do_break_episode():
            if self.problem.simulation_needs_action():
                s = self.problem.get_current_state()
                if s.any():
                    self.problem.pause_simulation()
                    # update q_values with state transition
                    if s_old is not None:
                        # get q-values

                        # update policy
                        self.policy.update_policy(log_prob, self.q_table[s_old][a])

                        r = self.problem.get_reward(s_old)
                        cum_reward += r
                        is_goal_state = self.problem.is_goal_state(s)
                        self.update_q_values(s_old, a, r, s, is_goal_state)

                        self.rpb.push(s_old, a, r)
                        #print(f"{steps}: \t {r}  \t {self.problem.evaluation} {s_old} {a}")

                        if is_goal_state:
                            return
                    a, log_prob = self.policy.get_action(s)
                    #a = self.problem.solve_state(s)
                    self.problem.act(a)

                    s_old = s
                    steps += 1
                    self.problem.unpause_simulation()

        print("Cum Reward: ", cum_reward, "Score: ", self.problem.evaluation)


