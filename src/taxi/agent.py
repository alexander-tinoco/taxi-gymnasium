import random
from pathlib import Path

import gymnasium as gym
import numpy as np

ALPHA = 0.9
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 10000
MAX_STEPS = 100
CHECKPOINT_EVERY = 1000
DEMO_EPISODES = 5

# Anchored to this file so the saved table is found no matter where the command runs.
Q_TABLE_PATH = Path(__file__).with_name("q_table_taxi.npy")


def train(env: gym.Env) -> np.ndarray:
    """Learn a Q-table with epsilon-greedy tabular Q-learning."""
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()

        for _ in range(MAX_STEPS):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, truncated, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            q_table[state, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)

            state = next_state
            if done or truncated:
                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Checkpoint so an interrupted run does not lose the progress.
        if (episode + 1) % CHECKPOINT_EVERY == 0:
            np.save(Q_TABLE_PATH, q_table)
            print(f"Episodio {episode + 1}/{NUM_EPISODES} - checkpoint guardado")

    np.save(Q_TABLE_PATH, q_table)
    print("Entrenamiento terminado y tabla guardada")
    return q_table


def watch(q_table: np.ndarray) -> None:
    """Replay the greedy policy of the Q-table in a pygame window."""
    env = gym.make("Taxi-v4", render_mode="human")

    for episode in range(DEMO_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        print("Episode", episode)

        for _ in range(MAX_STEPS):
            env.render()
            action = np.argmax(q_table[state, :])
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                env.render()
                print("Finished episode", episode, "with reward", total_reward)
                break

    env.close()


def main() -> None:
    if Q_TABLE_PATH.exists():
        q_table = np.load(Q_TABLE_PATH)
        print("Tabla Q cargada, se omite el entrenamiento")
    else:
        env = gym.make("Taxi-v4")
        q_table = train(env)
        env.close()

    watch(q_table)


if __name__ == "__main__":
    main()
