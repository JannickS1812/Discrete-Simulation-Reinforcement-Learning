import torch

from reinforce_agent import PolicyNetwork, ReinforceAgent
from deep_q_learning_agent import DeepQTable, DeepQLearningAgent


class Actor(PolicyNetwork):

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