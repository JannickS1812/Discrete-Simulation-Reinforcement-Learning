from vc_environment import Environment
from agents.deep_q_learning_agent import DeepQLearningAgent, DoubleDeepQLearningAgent
import matplotlib.pyplot as plt
import numpy as np

# set max number of iterations
max_iterations = 500
size = (2, 2)
it = 0
env = Environment(size)
agent = DoubleDeepQLearningAgent(env.problem, max_N_exploration=10, file="agents/q_table_single.npy", batch_size=10)
agent_prio = DoubleDeepQLearningAgent(env.problem, max_N_exploration=10, file="agents/q_table_prio.npy", batch_size=10, prioritized_replay=True)
performance = []
q_table = None
# training
while it < max_iterations:
    # standard DQN
    complexity = max(1, env.problem.eval(env.problem))
    q_table, N_sa = agent.train()
    energy_spent = env.problem.energy_spend
    perf = energy_spent/complexity
    env.reset()

    # Double DQN
    complexity = max(1, env.problem.eval(env.problem))
    q_table, N_sa = agent_prio.train()
    energy_spent = env.problem.energy_spend
    perf_double = energy_spent / complexity
    env.reset()

    performance.append((perf, perf_double))
    print(it, performance[it])
    it += 1

# plot results
performance = np.array(performance)
N = 50
plt.plot(performance[:, 0], 'b', alpha=.3)
plt.plot(performance[:, 1], 'r', alpha=.3)
plt.plot(np.convolve(performance[:, 0], np.ones(N)/N, mode='valid'), 'r', label='off')
plt.plot(np.convolve(performance[:, 1], np.ones(N)/N, mode='valid'), 'b', label='on')
plt.title('Prioritized Replay')
plt.legend()
plt.show()

# save q_table
agent.save_q_table()
agent_prio.save_q_table()
