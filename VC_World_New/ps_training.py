from ps_environment import Environment, SortingRobotPlantSimProblem
from agents.q_actor_critic import QActorCriticAgentSortingRobot
from agents.deep_q_learning_agent import DeepDuelingQTable
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
plantsim = Plantsim(version='16.1', license_type='Educational', visible = True, trust_models=True)
plantsim.load_model(r'C:\Users\Philipp\Documents\Uni\Diskrete Simulatzion und RL Projekt\Hartmann_Stranghoener_10_Abgabe_Projekt_3.spp')
plantsim.set_path_context('.Modelle.Modell')
plantsim.set_path_context('.Modelle.Modell')
plantsim.set_event_controller()

# set max number of iterations

max_iterations = 10000
it = 0
env = Environment(plantsim, problem_class=SortingRobotPlantSimProblem)
agent = QActorCriticAgentSortingRobot(env.problem, ValueNetworkClass=DeepDuelingQTable)
performance_train = []
q_table = None
# training
while it < max_iterations:
    print(it)
    it += 1
    agent.train()
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    env.reset()

# test_agent#
env = Environment(plantsim)
agent = QActorCriticAgentSortingRobot(env.problem)
performance_test = []
number_of_tests = 20
it = 0
while it < number_of_tests:
    it += 1
    while not env.problem.is_goal_state(env.problem):
        action = agent.act()
        if action is not None:
            env.problem.act(action)
    evaluation = env.problem.evaluation
    performance_test.append(evaluation)
    env.reset()

# plot results
x = np.array(performance_train)
N = int(max_iterations/10)
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance_train)
plt.plot(moving_average)
plt.show()

N = int(number_of_tests/10)
x = np.array(performance_test)
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance_test)
plt.plot(moving_average)
plt.show()

# save q_table
agent.save_q_table("agents/q_table.npy")
plantsim.quit()
