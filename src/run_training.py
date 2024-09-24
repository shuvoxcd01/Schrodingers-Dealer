from __future__ import annotations
import matplotlib.pyplot as plt
import gymnasium as gym
from src.algorithms.monte_carlo_es import MonteCarloES
from src.algorithms.monte_carlo_off_policy_control import MonteCarloOffPolicyControl
from src.util import create_grids, create_plots

n_episodes = 1000000
env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
agent = MonteCarloES(env=env, n_iter=n_episodes)
agent.run_policy_iteration()
agent.save_policy()


value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()


