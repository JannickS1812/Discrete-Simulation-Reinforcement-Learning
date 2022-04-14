from agent import Agent
import numpy as np


class SimpleReflexAgent(Agent):

    def __init__(self, problem, q_table=None):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = self.create_q_table()

    def act(self):
        # perception
        current_state = self.problem.get_current_state()
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def create_q_table(self):
        q_table = np.zeros((len(self.states), (len(self.actions))))
        # Put your source code here
        # w.g. q_table[0, 1] = 5 asserts a q_value of 5 to perform action 1 in state 0
        # the corresponding states and actions can be obtained by self.states[0] and
        # self.actions[1] in this example
        return q_table

