import os
import gymnasium as gym

from src.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from src.data import DATA_DIR

SAVED_POLICY_PATH = os.path.join(DATA_DIR, "MCES_saved_policy.pkl")

policy = BaseLearningAlgorithm.load_policy(saved_policy_path=SAVED_POLICY_PATH)

env = gym.make("Blackjack-v1", sab=True, render_mode="human")
num_wins = 0
num_draws = 0
num_losses = 0

for i in range(100):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        env.render()
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    print(f"Game: {i} Result:{reward}")

    if reward == 1:
        num_wins += 1
    elif reward == 0:
        num_draws += 1
    else:
        num_losses += 1

print("Total wins: ", num_wins)
print("Total draws: ", num_draws)
print("Total losses: ", num_losses)
