from ps_environment import Environment, SortingRobotPlantSimProblem
from agents.q_actor_critic import QActorCriticAgentPlantSim
from agents.deep_q_learning_agent import DeepQLearningAgentPlantSim, DeepQTable
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
#agent = QActorCriticAgentPlantSim(env.problem, ValueNetworkClass=DeepQTable)
agent = DeepQLearningAgentPlantSim(env.problem, batch_size=100)
performance_train = []
q_table = None
# training
cumsums = []
scores = []
cnt = 0
while it < max_iterations:
    print(f"Iteration {it}, P(random action)={0.9**it*100:.1f}%")
    it += 1
    cumsum, score = agent.train(random_action=0.9**it)
    if score == 100:
        cnt += 1
        if cnt > 5:
            it = max_iterations-5
            agent.save()
    else:
        cnt = 0

    cumsums.append(cumsum)
    scores.append(score)
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    env.reset()
    #plt.plot(cumsums)
    #plt.show(block=False)
    print(cumsums)
    print(scores)


# test_agent#
env = Environment(plantsim)
#agent = QActorCriticAgentPlantSim(env.problem, ValueNetworkClass=DeepQTable)
agent = DeepQLearningAgentPlantSim(env.problem)
agent.load()
performance_test = []
number_of_tests = 1
it = 0
max_steps = 500
while it < number_of_tests:
    it += 1

    steps = 0
    while steps < max_steps and not env.problem.is_goal_state(None):
        if env.problem.simulation_needs_action():
            s = env.problem.get_current_state()
            if s.any():
                env.problem.pause_simulation()

                filter = env.problem.filter_valid_actions(s)
                q_values = [q if filter[i] else -np.inf for i, q in enumerate(env.q_table[s])]
                a = self.actions[np.argmax(q_values)]

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
