from .agent import Agent
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

        size_x = self.problem.building.size[0]
        size_y = self.problem.building.size[1]
        for i, state in enumerate(self.states):
            pos_x = state[-2]
            pos_y = state[-1]

            if not state[ pos_x * size_y + pos_y]:  # dirty cell
                q_table[i, 0] = 1
            else:
                # choose one of the possible actions at random
                # this however does not avoid circles (up->down->up...)
                allowed_actions = []
                if pos_x < (size_x - 1):
                    allowed_actions.append(3)
                if pos_y < (size_y - 1):
                    allowed_actions.append(1)
                if pos_x > 1:
                    allowed_actions.append(2)
                if pos_y > 1:
                    allowed_actions.append(4)

                if not allowed_actions:  # no allowed actions, happens only for 1x1 world
                    continue

                q_table[i, np.random.choice(allowed_actions)] = 1

        return q_table

