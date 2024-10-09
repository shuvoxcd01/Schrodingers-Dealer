from flask import Flask, jsonify, render_template
from flask_caching import Cache
import random
import gymnasium as gym
from src.algorithms.monte_carlo_es import MonteCarloES
from src.util import create_grids, create_plots

app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache"})

env = gym.make("Blackjack-v1", sab=True)
agent = MonteCarloES(env=env)
ACTION_MAP = {0: "Stick", 1: "Hit"}
OBSERVATIONS = []
ACTIONS = []
REWARDS = []
CURRENT_EPISODE = 0


def get_card(card_value: int):
    suits = ["H", "D", "C", "S"]  # Hearts, Diamonds, Clubs, Spades
    values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    if card_value == 1:
        value = "A"
    elif card_value == 10:
        value = random.choice(["10", "J", "Q", "K"])
    else:
        value = str(card_value)

    card = value + random.choice(suits)  # e.g., '4H', 'AS'

    return card


@app.route("/")
def index():
    return render_template("index.html")  # Render the HTML template


@app.route("/data")
@cache.cached(timeout=5)
def get_data():
    global OBSERVATIONS, ACTIONS, REWARDS, CURRENT_EPISODE

    if not OBSERVATIONS:
        num_iter = 1
        OBSERVATIONS, ACTIONS, REWARDS = agent.run_policy_iteration(num_iter=num_iter)
        CURRENT_EPISODE += num_iter

    observation = OBSERVATIONS.pop(0)
    action = ACTIONS.pop(0)
    reward = REWARDS.pop(0)

    player_sum, dealer_face_card, is_usable_ace = observation

    card = get_card(dealer_face_card)
    action = ACTION_MAP[action]

    fig1_name, fig2_name = generate_figures()

    # Prepare the response data
    response = {
        "card": card,
        "total": player_sum,
        "action": action,
        "reward": reward,
        "episode_no": CURRENT_EPISODE,
        "image1": f"static/images/{fig1_name}.png",
        "image2": f"static/images/{fig2_name}.png",
    }

    return jsonify(response)


def generate_figures():
    fig1_name = "with_usable_ace"
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    create_plots(
        value_grid, policy_grid, title="With usable ace", figure_name=fig1_name
    )

    fig2_name = "without_usable_ace"
    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    create_plots(
        value_grid, policy_grid, title="Without usable ace", figure_name=fig2_name
    )
    return fig1_name, fig2_name


if __name__ == "__main__":
    app.run(debug=True)
