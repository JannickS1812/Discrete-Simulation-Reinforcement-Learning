from random import seed
import itertools
import numpy as np

from plantsim.plantsim import Plantsim
from problem import Problem


class PlantSimulationProblem(Problem):

    def __init__(self, plantsim: Plantsim, states=None, actions=None, id=None, evaluation=0, goal_state=False):
        """

        :param plantsim: Simulation model must include two tables "Actions" and "States" with both column and row index.
                         column index should be "Index" and "Name" for Actions, where "Name" contains the names of the
                         actions. For state the column index should be the names of the attributes. Also add "id" as
                         integer where the object id is saved, "evaluation" as a float value and "goal_state" as a
                         boolean value.
                         Also include two tables for the data exchange:
                         Table1: "CurrentState" with "id", one column for each state, "evaluation" and "goal_state"
                         Table2: "ActionControl" with "id",one column for each state and "action"

        """
        self.plantsim = plantsim
        if actions is not None:
            self.actions = actions
        else:
            self.actions = self.plantsim.get_object("Actions").get_columns_by_header("Name")
        if states is not None:
            self.states = states
        else:
            self.states = {}
            states = self.plantsim.get_object("States")
            for header in states.header:
                if header != "Index":
                    self.states[header] = states.get_columns_by_header(header)
                    # removing empty cells
                    self.states[header] = list(filter(None, self.states[header]))
        self.state = None
        self.id = id
        self.evaluation = evaluation
        self.goal_state = goal_state
        self.next_event = True

    def copy(self):
        ps_copy = PlantSimulationProblem(self.plantsim, self.state.copy(), self.actions[:], self.id, self.evaluation,
                                         self.goal_state)
        return ps_copy

    def act(self, action):
        self.plantsim.set_value("ActionControl[\"id\",1]", self.id)
        for label, values in self.states.items():
            for value in self.state:
                if value in values:
                    self.plantsim.set_value(f"ActionControl[\"{label}\",1]", value)
                    self.plantsim.set_value(f"ActionControl[\"{label}\",1]", value)
        self.plantsim.set_value("ActionControl[\"action\",1]", action)
        self.plantsim.execute_simtalk("AIControl")
        if not self.plantsim.plantsim.IsSimulationRunning():
            self.plantsim.start_simulation()

        self.next_event = True

    def to_state(self):
        return tuple(self.state)

    def is_goal_state(self, state):
        return state.goal_state

    def get_applicable_actions(self, state):
        return self.actions

    def get_current_state(self):
        """
        possible actions list named "actions" must be returned be simulation within the message
        :return:
        """
        if self.next_event:
            self.state = []
            #states = self.plantsim.get_next_message()
            states = self.plantsim.get_current_state()
            for key, value in states.items():
                if key == "id":
                    self.id = value
                elif key == "evaluation":
                    self.evaluation = value
                elif key == "goal_state":
                    self.goal_state = value
                else:
                    self.state.append(value)
            self.next_event = False
        return self

    def eval(self, state):
        return state.evaluation

    def get_all_actions(self):
        return self.actions

    def get_all_states(self):
        all_states = list(itertools.product(*list(self.states.values())))
        all_states = [tuple(x) for x in all_states]
        return all_states

    def get_reward(self, state):
        reward = -self.eval(state)
        return reward

    def reset(self):
        self.state = None
        self.id = None
        self.evaluation = 0
        self.goal_state = False
        self.next_event = True


class SortingRobotPlantSimProblem(Problem):
    def __init__(self, plantsim):
        self.plantsim = plantsim
        self.plantsim.set_event_controller()

    @property
    def evaluation(self):
        return self.__getScore()

    def reset(self):
        '''
        resets simulation
        '''
        self.plantsim.reset_simulation()
        self.plantsim.set_value(r'RL_Agent_Interaction["Score", 1]', 0)

    def start(self):
        '''
        starts simulation
        '''
        self.plantsim.start_simulation()

    def pause_simulation(self):
        '''
        pauses simulation for prediction and calculation
        '''
        self.plantsim.execute_simtalk(r'Ereignisverwalter.stop()')

    def unpause_simulation(self):
        '''
        unpauses simulation for prediction and calculation
        '''
        self.plantsim.execute_simtalk(r'Ereignisverwalter.start()')

    def is_goal_state(self, state):
        '''
        is in goal state, if all elements (100) are sorted correctly

        '''
        return self.__getScore() == 100

    def is_action_valid(self, action_pull, state, write_to_plantsim_if_false):
        '''
        returns true if there is a product to pull from (determined by action_pull)

        write_to_plantsim_if_false: is necessary, because if if action is not valid
        plantsimulation will not know about the action.
        To be able to obtain a reward fro the action the reward will be written manually to plantsim

        '''
        #conv_onehot, buf_onehot, type1_onehot, type2_onehot, amount1, amount2 = state
        if action_pull == 0: #Pull from Conveyer
            if state[0] == 1: #Conveyer is empty
                if write_to_plantsim_if_false:
                    self.__write_reward(-1000)
                return False
        elif action_pull == 1: #Pull from buffer
            if state[4] == 1: #buffer is empty
                if write_to_plantsim_if_false:
                    self.__write_reward(-1000)
                return False
        return True

    def get_all_states(self):
        '''
        Returns numpy array of all possible states
        '''
        conv_input_type     = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        buffer_input_type   = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        storage1_type       = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        storage2_type       = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        storage1_amount     = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        storage2_amount     = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

        all_states = [[c, b, s1, s2, s1m, s2m]
                      for c in conv_input_type
                      for b in buffer_input_type
                      for s1 in storage1_type
                      for s2 in storage2_type
                      for s1m in storage1_amount
                      for s2m in storage2_amount]

        all_states = np.array([item
                      for l in all_states
                      for sublist in l
                      for item in sublist])
        return all_states


    def get_current_state(self):
        '''
        Returns current state as 1D int np.array
        State consists of
        conveyer belt input onehot eg [0,0,0,1]
        buffer input onehot eg [0,0,1,0]
        type of products in storage1 onehot eg [0,0,1,0]
        type of products in storage2 onehot eg [1,0,0,0]
        amount of same products in storage1 onehot eg 2
        amount of same products in storage2 onehot eg 9
        everything is concatenated into 1D np.array eg.:
        [0,0,0,1], [0,0,1,0], [0,0,1,0], [1,0,0,0], 2, 9   results into:
        [0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,2,9]
        this numpy array will be returned
        '''

        #Get Inputs from conveyer Belt and buffer
        r = self.__getBufferAndConveyerBeltInformation()
        if r:
            conv_onehot, buf_onehot = r
        else:
            return np.array(0)

        #Get Inputs from Storages
        type1_onehot, type2_onehot, amount1, amount2 = self.__getStorageInformation()

        l = [conv_onehot.tolist(), buf_onehot.tolist(), type1_onehot.tolist(), type2_onehot.tolist(), [amount1], [amount2]]
        s = np.array([item for sublist in l for item in sublist])
        return s

    def get_all_actions(self):
        '''
        Get all possible actions (action pairs)
        (Pull, Target)
        Pull: From where to take material (0=Conveyer of Production; 1=Buffer of Cycle)
        Target: Where to place the pulled material (0=Storage1; 1=Storage2; 2=Cycle)
        '''
        return range(6)

    def get_reward(self):
        # reward from last action
        reward = self.plantsim.get_value(r'RL_Agent_Interaction["RewardFromLastAction", 1]') - 0.1  # -0.1 penalty for every step
        return reward

    def act(self, action):
        '''
        Action has to be a tuple or a list of integers: (Pull, Target)
        Pull: From where to take material (0=Conveyer of Production; 1=Buffer of Cycle)
        Target: Where to place the pulled material (0=Storage1; 1=Storage2; 2=Cycle)

        Writing Action into plantsim table and then calling the processing of it
        '''
        self.__setPull(action // 3) #ziehen: 0 Förderstrecke     1 Puffer
        self.__setTarget(action % 2) #target: 0 Lager1     1 Lager2     2 Puffer
        self.plantsim.execute_simtalk(r'AI_DoAction()')

    def simulation_needs_action(self):
        '''
        Evaluates if plantsimulation is waiting for a decision of the agent
        check if both actions are -1 which means that the robot is waiting for an action
        '''
        return self.__getPull() == -1 == self.__getTarget()

    def filter_valid_actions(self, state):
        conv = state[0] != 1
        buf = state[4] != 1
        if conv == 0 == buf:
            print("No action valid")
        return [conv, conv, conv, buf, buf, buf]

    def quit_(self):
        '''
        quits the plantsim program
        '''
        self.plantsim.quit()
        print("Exited Plantsimulation Model")

    def __getBufferAndConveyerBeltInformation(self):
        '''
        Returns information about what is on which Input of the pick and place robot
        Conveyer --> Input from production line
        buffer --> Input from Cycle loop
        '''
        #Get Inputs from conveyer Belt and buffer
        conveyer_belt = self.plantsim.get_value(r'Förderstrecke19.Inhalt')
        if conveyer_belt != None:
            conveyer_belt = self.plantsim.get_value(r'Förderstrecke19.Inhalt.Inhalt')
        buffer = self.plantsim.get_value(r'Puffer.Inhalt')
        if buffer != None:
            buffer = self.plantsim.get_value(r'Puffer.Inhalt.Inhalt')


        conv_onehot = self.__class_to_onehot(conveyer_belt)

        buf_onehot = self.__class_to_onehot(buffer)

        if conveyer_belt == None == buffer:
            return False
        return conv_onehot, buf_onehot


    def __getStorageInformation(self):
        '''
        Returns information about what is in which storage and how many
        '''
        type1 = self.plantsim.get_value(r'RL_Agent_Interaction["Lager1Typ", 1]')
        type2 = self.plantsim.get_value(r'RL_Agent_Interaction["Lager2Typ", 1]')
        amount1 = self.plantsim.get_value(r'RL_Agent_Interaction["Lager1Menge", 1]')
        amount2 = self.plantsim.get_value(r'RL_Agent_Interaction["Lager2Menge", 1]')

        type1_onehot = self.__class_to_onehot(type1)
        type2_onehot = self.__class_to_onehot(type2)

        return type1_onehot, type2_onehot, amount1, amount2

    def __getPull(self):
        '''
        Gets the encoded target where PickAndPlace Robot should pick the material from
        0 or 1
        this value is -1 if robot asks for target
        '''
        return self.plantsim.get_value(r'RL_Agent_Interaction["Ziehen", 1]')

    def __setPull(self, val):
        '''
        0 or 1 or 2
        Sets the target to select material from if PickAndPlace Robot has no material
        '''
        self.plantsim.set_value(r'RL_Agent_Interaction["Ziehen", 1]', val)

    def __getTarget(self):
        '''
        0 or 1 or 2
        Gets the encoded target where PickAndPlace Robot should transport the material to
        this value is -1 if robot asks for target
        '''
        return self.plantsim.get_value(r'RL_Agent_Interaction["Ziel", 1]')

    def __setTarget(self, val):
        '''
        0 or 1 or 2
        Sets the target if PickAndPlace Robot has material
        '''
        self.plantsim.set_value(r'RL_Agent_Interaction["Ziel", 1]', val)

    def __getScore(self):
        '''
        Returns the score of the plantsimulation
        (the amount of materials which have been delivered)
        '''
        return self.plantsim.get_value(r'RL_Agent_Interaction["Score", 1]')

    def __class_to_onehot(self, cl):
        '''
        RETURNS One hot encoded NP Array for classes:
        0. No Material
        1. English Material
        2. Spanish Material
        3. German Material
        '''
        buf_onehot = np.zeros((4), dtype=np.int32)
        if cl == None:
            buf_onehot[0] = 1
        elif "Englisch" in cl:
            buf_onehot[1] = 1
        elif "Deutsch" in cl:
            buf_onehot[2] = 1
        elif "Spanisch" in cl:
            buf_onehot[3] = 1
        return buf_onehot


class Environment:

    def __init__(self, plantsim: Plantsim, problem_class=PlantSimulationProblem, seed_value=1):

        if seed_value is not None:
            seed(seed_value)
        plantsim.reset_simulation()
        self.problem = problem_class(plantsim)
        plantsim.start_simulation()

    def reset(self):
        self.problem.plantsim.execute_simtalk("reset")
        self.problem.plantsim.reset_simulation()
        self.problem.reset()
        self.problem.plantsim.start_simulation()