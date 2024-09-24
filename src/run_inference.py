import gymnasium as gym

from src.algorithms.base_learning_algorithm import BaseLearningAlgorithm

SAVED_POLICY_PATH = "/Users/falguni/Study/Repositories/GitHub/Schrodingers-Dealer/src/data/MCES_saved_policy.pkl"

policy = BaseLearningAlgorithm.load_policy(saved_policy_path=SAVED_POLICY_PATH)

env = gym.make("Blackjack-v1", sab=True, render_mode="human")
num_wins = 0

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

    win = 1 if reward == 1 else 0
    num_wins += win

print("Total wins: ", num_wins)