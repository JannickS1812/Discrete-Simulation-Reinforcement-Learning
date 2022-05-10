from .agent import Agent
import numpy as np
import copy


class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=15, R_Max=100):
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

    def argmax_with_random_choice_on_eq(self, arr):
        # returns the argmax of 'arr'
        # in case of tie returns a random index from the maximal values

        idc = np.where(arr == np.max(arr))[0]
        return np.random.choice(idc)

    def train(self):

        alpha = 0.1
        s_prev = a_prev = r_prev = None
        exploration_func = lambda u, n: self.R_Max if n < self.max_N_exploration else u

        # update R_Max to reflect the highest attainable reward for the given room size
        s_opt = copy.deepcopy(self.problem.get_current_state())
        s_opt.building.rooms = np.zeros(s_opt.building.rooms.shape).astype(int)
        self.R_Max = self.problem.eval(s_opt)

        while not self.problem.is_goal_state(self.problem):

            # current state, reward
            s = self.states.index(self.problem.get_current_state().to_state())
            r = self.problem.get_reward(self.problem.get_current_state())

            # update frequency- and Q-table
            if s_prev is not None:
                self.N_sa[s_prev, a_prev] += 1
                self.q_table[s_prev, a_prev] = self.q_table[s_prev, a_prev] + alpha * ((r - r_prev) + self.gamma * np.max(self.q_table[s, :]) - self.q_table[s_prev, a_prev])

            # get highest exploration function value
            expl_func_vals = [exploration_func(self.q_table[s, a], self.N_sa[s, a]) for a in range(len(self.actions))]
            action = self.actions[self.argmax_with_random_choice_on_eq(expl_func_vals)] # in case of tie take random action
            a = self.actions.index(action)

            # act and update running state vars
            self.problem.act(action)
            s_prev = s
            a_prev = a
            r_prev = r

        return self.q_table, self.N_sa

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)
