from .agent import Agent
import numpy as np


class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((len(self.states), (len(self.actions))))
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = np.zeros((len(self.states), (len(self.actions))))
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):

        exploration_func = lambda u, n: self.R_Max if n < self.max_N_exploration else u

        while not self.problem.is_goal_state(self.problem):
            #if np.random.random() < 0.2:
            #    action = np.random.choice(self.problem.get_applicable_actions(self.problem.get_current_state()))
            #else:
            #    action = self.act()



            s = self.states.index(self.problem.get_current_state().to_state())
            r = self.problem.get_reward(self.problem.get_current_state())
            action = self.actions[np.argmax([exploration_func(self.q_table[s, a], self.N_sa[s, a]) for a in range(len(self.actions))])]

            self.problem.act(action)

            s_new = self.states.index(self.problem.get_current_state().to_state())
            r_new = self.problem.get_reward(self.problem.get_current_state())
            a = self.actions.index(action)

            self.N_sa[s_new, a] += 1
            self.q_table[s, a] = self.q_table[s, a] + 0.1 * ((r_new - r) + self.gamma * np.max(self.q_table[s_new, :]) - self.q_table[s, a])

        return self.q_table, []

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)
