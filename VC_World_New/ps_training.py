from ps_environment import Environment, SortingRobotPlantSimProblem
from agents.q_actor_critic import QActorCriticAgentPlantSim
from agents.deep_q_learning_agent import DeepQLearningAgentPlantSim, DeepQTable
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
plantsim = Plantsim(version='16.1', license_type='Educational', visible = True, trust_models=True)
plantsim.load_model(r'C:\Users\Philipp\Documents\Uni\Diskrete Simulatzion und RL Projekt\Hartmann_Stranghoener_10_Abgabe_Projekt_3.spp')
plantsim.set_path_context('.Modelle.Modell')
plantsim.set_path_context('.Modelle.Modell')
plantsim.set_event_controller()


env = Environment(plantsim, problem_class=SortingRobotPlantSimProblem)
best_time = 20000 #time in s
all_times = [best_time]
all_times_x = [1]
agent = DeepQLearningAgentPlantSim(env.problem, prioritized_replay=False, batch_size=50, trainings_per_step = 1)
agent.load()
def eval_model(iteration):
    """
    returns true if model is better than anything previous
    """
    global best_time
    avg_rew_per_step, score, solve_time = agent.eval()
    if score == 100 and solve_time < best_time:
        best_time = solve_time
        agent.save(str(solve_time))
    print(
        f"Evaluation of model: Solved Problem: {100 == score}, Time to solve or exit: {solve_time:.1f}s, Best Solve Time: {best_time:.1f}s")
    all_times_x.append(iteration)
    all_times.append(best_time)
    env.reset()
    return best_time


q_table = None
avg_rew_per_step = []
scores = []
cnt = 0

max_iterations = 250
p = 0.98 #epsilon parameter
it = 0
while it < max_iterations:
    start = time.time()
    print(f"Iteration {it}, P(random action)={p**it*100:.1f}%")

    rew, score = agent.train(random_action=p**it)
    it += 1
    agent.save()

    avg_rew_per_step.append(rew)
    scores.append(score)
    env.reset()

    end = time.time()
    print(f"Average Reward per Step: {avg_rew_per_step[-1]:.1f}, Achieved Score: {scores[-1]} Epoch Time: {end - start:.1f}s")

    if score > 80:
        eval_model(it)



# plot results
x = np.array(avg_rew_per_step)
#N = int(max_iterations/10)
#moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.figure(0)
plt.title("Average Reward per Step")
plt.xlabel("Epoch")
plt.ylabel("Average Reward per Step")
plt.plot(avg_rew_per_step)
#plt.plot(moving_average)



plt.figure(1)
plt.title("Achieved Score")
plt.xlabel("Epoch")
plt.ylabel("Achieved Score")
plt.plot(scores)



plt.figure(2)
plt.title("Time to solve Problem [seconds] (20.000 = unsolved)")
plt.xlabel("Epoch")
plt.ylabel("Best achieved model")
plt.plot(all_times_x, all_times)


plt.show()

plantsim.quit()
