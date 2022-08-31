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
    
    
    

class Agent():
    def __init__(self):
        return
    def act(self, state):
        """
        receives state 
        returns action
        """
        #ziehen: 0 Förderstrecke     1 Puffer
        #target: 0 Lager1     1 Lager2     2 Puffer
        conv_onehot, buf_onehot, type1_onehot, type2_onehot, amount1, amount2 = state

        pull = np.random.choice([0,1],1)[0]
        target = np.random.choice([0,1,2],1)[0]


        return pull, target
    
    def train():
        
        pass
        




#ziehen: 0 Förderstrecke     1 Puffer
#Ziel  : 0 Lager1  1 Lager2  2 Puffer
if __name__ == "__main__":
    a = Agent()
    psc = PlantSimConnector()
    try:
        psc.reset()
        psc.start()
        psc.doEpoch(a)
    except KeyboardInterrupt: #Close PlantsimWindow if KeyboardInterrupt
        psc.quit_()
        print('Interrupted')



        





