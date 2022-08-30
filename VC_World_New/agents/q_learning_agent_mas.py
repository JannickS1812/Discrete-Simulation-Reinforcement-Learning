from agent import Agent
import numpy as np


class QLearningAgentMAS(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=50, R_Max=500):
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
        if self.problem.is_goal_state(current_state):
            return None
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        # states and action are valid only for an autonomous agent with a certain "id"
        # thus states und actions are saved in dictionaries with the id as key
        action = None
        actions = {}
        states = {}
        while True:
            current_state = self.problem.get_current_state()
            r = self.problem.get_reward(current_state)
            if current_state.id in states and current_state.id in actions:
                s = states[current_state.id]
                action = actions[current_state.id]
            else:
                s = None
                action = None
            s_new = self.states.index(current_state.to_state())
            states[current_state.id] = s_new
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s, a] += 1
                self.q_table[s, a] = self.q_table[s, a] + self.alpha(s, a) * (r + self.gamma * np.max(self.q_table[s_new]) - self.q_table[s, a])
            if self.problem.is_goal_state(current_state):
                return self.q_table, self.N_sa

            action = self.choose_GLIE_action(self.q_table[s_new], self.N_sa[s_new])
            actions[current_state.id] = action
            # act
            self.problem.act(action)

    def choose_GLIE_action(self, q_values, N_s):
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = (self.R_Max / 2 + q_values)
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not divide by zero

        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values)
        else:
            probabilities = np.copy(max_values)
        # set not possible actions to zero

        # norming
        probabilities = probabilities / probabilities.sum()
        # select action according to the (q) values
        if np.sum(no_sufficient_exploration) > 0:
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action = self.actions[np.argmax(probabilities)]
        return action

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)

    def alpha(self, s, a):
        # learning rate alpha decreases with N_sa for convergence
        alpha = self.N_sa[s, a] ** (-1 / 2)
        return alpha
