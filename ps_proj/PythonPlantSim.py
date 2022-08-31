# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:43:47 2022

@author: Philipp
"""

from plantsim.plantsim import Plantsim
from plantsim.table import Table
import plantsim
import numpy as np
import atexit

import sys
import os



class Problem:
    def act(self, action):
        """
        Peforms the action passed
        """
        pass

    def is_goal_state(self, state):
        """
        Checks if state is a goal state
        :param state: Problem
        :return: Boolean
        """
        pass

    def is_action_valid(self, action, state):
        """
        Returns a list of actions applicable in state
        :param state: Problem
        :return: list<String>
        """
        actions = []
        return actions

    def get_current_state(self):
        """
        returns itself and eventually performs an update first
        :return:
        """
        return self


    def get_all_actions(self):
        """
        returns a list of all actions
        :return: list<string>
        """
        actions = []
        return actions

    #def get_all_states(self):
    #    """
    #    returns a list of all states
    #    :return: list<string>
    #    """
    #    states = []
    #    return states

    def get_reward_from_last_action(self):
        """
        Calulates a reward of the state for RL
        :param state: Problem
        :return: float
        """
        r = 0
        return r

    def reset(self):
        """
        resets the environment
        """
        pass




class PlantSimProblem_SortingRobot(Problem):
    def __init__(self):
        self.ps = Plantsim(version='16.1', license_type='Educational', visible = True, trust_models=True)
        self.ps.load_model(r'C:\Users\Philipp\Documents\Uni\Diskrete Simulatzion und RL Projekt\Hartmann_Stranghoener_10_Abgabe_Projekt_3.spp')
        self.ps.set_path_context('.Modelle.Modell')
        self.ps.set_event_controller()
    
    def reset(self):
        '''
        resets simulation 
        '''
        self.ps.reset_simulation()
        self.ps.set_value(r'RL_Agent_Interaction["Score", 1]', 0)
        
    def start(self):
        '''
        starts simulation 
        '''
        self.ps.start_simulation()
    
    def pause_simulation(self):
        '''
        pauses simulation for prediction and calculation
        '''
        self.ps.execute_simtalk(r'Ereignisverwalter.stop()')
        
    def unpause_simulation(self):
        '''
        unpauses simulation for prediction and calculation
        '''
        self.ps.execute_simtalk(r'Ereignisverwalter.start()')
        
    def is_goal_state(self):
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
    

        
        return type1_onehot, type2_onehot, amount1, amount2
    
    def get_all_actions(self):
        '''
        Get all possible actions (action pairs)
        (Pull, Target)
        Pull: From where to take material (0=Conveyer of Production; 1=Buffer of Cycle)
        Target: Where to place the pulled material (0=Storage1; 1=Storage2; 2=Cycle)
        '''
        pulls = range(2)
        targets = range(3)
        actions = [[pull, target] for pull in pulls for target in targets]
        return actions
    
    def get_reward_from_last_action(self):
        #reward from last action
        reward = self.ps.get_value(r'RL_Agent_Interaction["RewardFromLastAction", 1]') - 0.1 # -0.1 penalty for every step
        return reward
    
    def act(self, action):
        '''
        Action has to be a tuple or a list of integers: (Pull, Target)
        Pull: From where to take material (0=Conveyer of Production; 1=Buffer of Cycle)
        Target: Where to place the pulled material (0=Storage1; 1=Storage2; 2=Cycle)
        
        Writing Action into plantsim table and then calling the processing of it
        '''
        self.__setPull(action[0]) #ziehen: 0 Förderstrecke     1 Puffer
        self.__setTarget(action[1]) #target: 0 Lager1     1 Lager2     2 Puffer
        self.ps.execute_simtalk(r'AI_DoAction()')
        
    def simulation_needs_action(self):
        '''
        Evaluates if plantsimulation is waiting for a decision of the agent
        check if both actions are -1 which means that the robot is waiting for an action
        '''
        return self.__getPull() == -1 == self.__getTarget()
    
    def quit_(self):
        '''
        quits the plantsim program
        '''
        self.ps.quit()
        print("Exited Plantsimulation Model")
        
    def __write_reward(self, r):
        '''
        is used if an action is not valid
        action does not get sent to Plantsim
        The reward is manually writte n into the rewardfromlastaction column
        '''
        self.ps.set_value(r'RL_Agent_Interaction["RewardFromLastAction", 1]', r)
    
    def __getBufferAndConveyerBeltInformation(self):
        '''
        Returns information about what is on which Input of the pick and place robot
        Conveyer --> Input from production line
        buffer --> Input from Cycle loop
        '''
        #Get Inputs from conveyer Belt and buffer
        conveyer_belt = self.ps.get_value(r'Förderstrecke19.Inhalt')
        if conveyer_belt != None:
            conveyer_belt = self.ps.get_value(r'Förderstrecke19.Inhalt.Inhalt')
        buffer = self.ps.get_value(r'Puffer.Inhalt')
        if buffer != None:
            buffer = self.ps.get_value(r'Puffer.Inhalt.Inhalt')
            
        
        conv_onehot = self.__class_to_Onehot(conveyer_belt)
            
        buf_onehot = self.__class_to_Onehot(buffer)
        
        if conveyer_belt == None == buffer:
            return False
        return conv_onehot, buf_onehot
        
        
    def __getStorageInformation(self):
        '''
        Returns information about what is in which storage and how many
        '''
        type1 = self.ps.get_value(r'RL_Agent_Interaction["Lager1Typ", 1]')
        type2 = self.ps.get_value(r'RL_Agent_Interaction["Lager2Typ", 1]')
        amount1 = self.ps.get_value(r'RL_Agent_Interaction["Lager1Menge", 1]')
        amount2 = self.ps.get_value(r'RL_Agent_Interaction["Lager2Menge", 1]')
        
        type1_onehot = self.__class_to_Onehot(type1)
        type2_onehot = self.__class_to_Onehot(type2)
        
        return type1_onehot, type2_onehot, amount1, amount2
    
    def __getPull(self):
        '''
        Gets the encoded target where PickAndPlace Robot should pick the material from
        0 or 1
        this value is -1 if robot asks for target
        '''
        return self.ps.get_value(r'RL_Agent_Interaction["Ziehen", 1]')
    
    def __setPull(self, val):
        '''
        0 or 1 or 2
        Sets the target to select material from if PickAndPlace Robot has no material
        '''
        self.ps.set_value(r'RL_Agent_Interaction["Ziehen", 1]', val)

    def __getTarget(self):
        '''
        0 or 1 or 2
        Gets the encoded target where PickAndPlace Robot should transport the material to
        this value is -1 if robot asks for target
        '''
        return self.ps.get_value(r'RL_Agent_Interaction["Ziel", 1]')
    
    def __setTarget(self, val):
        '''
        0 or 1 or 2
        Sets the target if PickAndPlace Robot has material
        '''
        self.ps.set_value(r'RL_Agent_Interaction["Ziel", 1]', val)
        
    def __getScore(self):
        '''
        Returns the score of the plantsimulation
        (the amount of materials which have been delivered)
        '''
        return self.ps.get_value(r'RL_Agent_Interaction["Score", 1]')
    
    def __class_to_Onehot(self, cl):
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
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
class Controller:
    def __init__(self, plantsimworld: PlantSimProblem_SortingRobot, agent):
        self.psw :PlantSimProblem_SortingRobot = plantsimworld
        self.agent = agent
        self.psw.get_all_states()
        
    def doEpoch(self):
        self.psw.reset()
        self.psw.start()
        self.psw.unpause_simulation()
        
        step = 0
        max_steps = 1000
        state_before = np.array([0])
        action_before = None
        
        while not self.psw.is_goal_state():
            '''Check if Plantsimulation is waiting for an action'''
            if self.psw.simulation_needs_action():
                
                '''Obtain State from PlantSim'''
                state = self.psw.get_current_state() #returns [0] if state is not valid -> else normal state
                if state.any(): #if state valid
                    self.psw.pause_simulation()
                    
                    '''Obtain Reward from last action'''
                    reward_from_last_action = self.psw.get_reward_from_last_action()
                    
                    '''Push to Replay Buffer if State is ready'''
                    if state_before.any():
                        #print("Step:", step, "Reward: ", reward_from_last_action, "Action: ", action_before, "State: ", state_before,  "valid: ", self.psw.is_action_valid(action_before[0], state_before, False))
                        #ReplayBuffer.push(state_before, reward_from_last_action)
                        pass
                    
                    '''Obtain action for current state and act'''
                    action_pull, action_target = self.agent.act(state)
                    if self.psw.is_action_valid(action_pull, state, True):
                        self.psw.act((action_pull, action_target))
                        
                    self.psw.unpause_simulation()
                    
                    '''Train agent'''
                    #self.agent.train()
    
                    state_before = state
                    action_before = (action_pull, action_target)

                    step+=1
                    if step == max_steps:
                        print("Max Steps reached. Break Simulation")
                        break
        

class Agent():
    def __init__(self):
        return
    def act(self, state):
        """
        receives state (1d np.array)
        returns action (tuple of (pull, target))
            Pull: From where to take material (0=Conveyer of Production; 1=Buffer of Cycle)
            Target: Where to place the pulled material (0=Storage1; 1=Storage2; 2=Cycle)
        """

        

        pull = np.random.choice([0,1],1)[0]
        target = np.random.choice([0,1,2],1)[0]


        return pull, target
    
    def train():
        
        pass
        





if __name__ == "__main__":
    agent = Agent()
    
    plantsim_problem = PlantSimProblem_SortingRobot()
    
    c = Controller(plantsim_problem, agent)
    try:
        for i in range(10):
            c.doEpoch()
    except (KeyboardInterrupt): #Close Plantsim at interrupt
        plantsim_problem.quit_()
        print('Interrupted')
        
        
 
        
 
    
 
    
 
    
 
    
 
    
    '''
    OLD VERSION 
    psc = PlantSimConnector()
    try:
        psc.reset()
        psc.start()
        psc.doEpoch(a)
    except KeyboardInterrupt: #Close PlantsimWindow if KeyboardInterrupt
        psc.quit_()
        print('Interrupted')
    '''



'''
OLD Programm
Not necessary any more, but it should still work
'''

        
class PlantSimConnector():
    def __init__(self):
        self.ps = Plantsim(version='16.1', license_type='Educational', visible = True, trust_models=True)
        self.ps.load_model(r'C:\Users\Philipp\Documents\Uni\Diskrete Simulatzion und RL Projekt\Hartmann_Stranghoener_10_Abgabe_Projekt_3.spp')
        self.ps.set_path_context('.Modelle.Modell')
        self.ps.set_event_controller()
        
        #self.states = states
        #self.actions = actions
        
    def reset(self):
        self.ps.reset_simulation()
        self.ps.set_value(r'RL_Agent_Interaction["Score", 1]', 0)
        
    def start(self):
        self.ps.start_simulation()
    
    def pauseSimulation(self):
        self.ps.execute_simtalk(r'Ereignisverwalter.stop()')
        
    def unpauseSimulation(self):
        self.ps.execute_simtalk(r'Ereignisverwalter.start()')
        
    def isFinished(self):
        return self.ps.get_value(r'RL_Agent_Interaction["Score", 1]') == 100
    
    def getScore(self):
        return self.ps.get_value(r'RL_Agent_Interaction["Score", 1]')
    
    def getPull(self):
        return self.ps.get_value(r'RL_Agent_Interaction["Ziehen", 1]')
    
    def setPull(self, val):
        self.ps.set_value(r'RL_Agent_Interaction["Ziehen", 1]', val)
    
            
    def getTarget(self):
        return self.ps.get_value(r'RL_Agent_Interaction["Ziel", 1]')
    
    def setTarget(self, val):
        self.ps.set_value(r'RL_Agent_Interaction["Ziel", 1]', val)
    
    def doAction(self, action_pull, action_target):
        self.setPull(action_pull) #ziehen: 0 Förderstrecke     1 Puffer
        self.setTarget(action_target) #target: 0 Lager1     1 Lager2     2 Puffer
        self.ps.execute_simtalk(r'AI_DoAction()')
    
    def Class_to_Onehot(self, cl):
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
        
    def get_state(self):
        
        #Get Inputs from conveyer Belt and buffer
        r = self.getBufferAndConveyerBeltInformation()
        if r:
            conv_onehot, buf_onehot = r
        else: 
            return False
            
        #Get Inputs from Storages
        type1_onehot, type2_onehot, amount1, amount2 = self.getStorageInformation()


        return conv_onehot, buf_onehot, type1_onehot, type2_onehot, amount1, amount2
    
    def getBufferAndConveyerBeltInformation(self):
        #Get Inputs from conveyer Belt and buffer
        conveyer_belt = self.ps.get_value(r'Förderstrecke19.Inhalt')
        if conveyer_belt != None:
            conveyer_belt = self.ps.get_value(r'Förderstrecke19.Inhalt.Inhalt')
        buffer = self.ps.get_value(r'Puffer.Inhalt')
        if buffer != None:
            buffer = self.ps.get_value(r'Puffer.Inhalt.Inhalt')
            
        
        conv_onehot = self.Class_to_Onehot(conveyer_belt)
            
        buf_onehot = self.Class_to_Onehot(buffer)
        
        if conveyer_belt == None == buffer:
            return False
        return conv_onehot, buf_onehot
        
        
    def getStorageInformation(self):
        type1 = self.ps.get_value(r'RL_Agent_Interaction["Lager1Typ", 1]')
        type2 = self.ps.get_value(r'RL_Agent_Interaction["Lager2Typ", 1]')
        amount1 = self.ps.get_value(r'RL_Agent_Interaction["Lager1Menge", 1]')
        amount2 = self.ps.get_value(r'RL_Agent_Interaction["Lager2Menge", 1]')
        
        type1_onehot = self.Class_to_Onehot(type1)
        type2_onehot = self.Class_to_Onehot(type2)
        
        return type1_onehot, type2_onehot, amount1, amount2
        
    def getReward(self):
        #reward from last action
        reward = self.ps.get_value(r'RL_Agent_Interaction["RewardFromLastAction", 1]')
        return reward
        
    def isActionValid(self, action_pull, state):
        conv_onehot, buf_onehot, type1_onehot, type2_onehot, amount1, amount2 = state
        if action_pull == 0: #Pull from Conveyer
            if conv_onehot[0] == 1: #Conveyer is empty
                return False
        elif action_pull == 1: #Pull from buffer
            if buf_onehot[0] == 1: #buffer is empty
                return False
        return True
    
    def writeReward(self, r):
        self.ps.set_value(r'RL_Agent_Interaction["RewardFromLastAction", 1]', r)
        
    def simulationNeedsAction(self):
        return self.getPull() == -1 == self.getTarget()
    
    def doEpoch(self, agent):
        self.reset()
        self.start()
        self.unpauseSimulation()
        
        step = 0
        max_steps = 10000
        state_before = None
        action_before = None
        while not self.isFinished():
            
            if self.simulationNeedsAction():
                state = self.get_state()
                if state:
                    self.pauseSimulation()
                    
                    reward_from_last_action = self.getReward() - 0.1 # -0.1 penalty for every step
                    
                    
                    if state_before and action_before:
                        print("Step:", step, "Reward: ", reward_from_last_action, "Action: ", action_before, "State: ", state_before,  "valid: ", self.isActionValid(action_before[0], state_before))
                        #ReplayBuffer.push(state_before, reward_from_last_action)
                        pass
                    
                    action_pull, action_target = agent.act(state)
                    if self.isActionValid(action_pull, state):
                        self.doAction(action_pull, action_target)
                    else:
                        self.writeReward(-1000)
                        
                    self.unpauseSimulation()
                        
                    #agent.train()
    
                    state_before = state
                    action_before = (action_pull, action_target)
                    print(self.getScore())
                    
                    step+=1
                    if step == max_steps:
                        print("Max Steps reached. Break Simulation")
                        break
    
    def quit_(self):
        self.ps.quit()
        print("Exited Plantsimulation Model")





