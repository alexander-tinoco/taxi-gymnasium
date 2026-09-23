import os
import random

import gymnasium as gym
import numpy as np

alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

archivo = "q_table_taxi.npy"

env = gym.make("Taxi-v4")


def choose_action(state):
    if random.uniform(a=0, b=1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


if os.path.exists(archivo):
    q_table = np.load(archivo)
    print("Tabla Q cargada, se omite el entrenamiento")
else:
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        for step in range(max_steps):
            action = choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            state = next_state

            if done or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Checkpoint cada 1000 episodios por si el programa se interrumpe
        if (episode + 1) % 1000 == 0:
            np.save(archivo, q_table)
            print(f"Episodio {episode + 1}/{num_episodes} - checkpoint guardado")

    np.save(archivo, q_table)
    print("Entrenamiento terminado y tabla guardada")

env.close()

env = gym.make("Taxi-v4", render_mode="human")

for episode in range(5):
    state, _ = env.reset()
    done = False
    total_reward = 0
    print("Episode", episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

        if done or truncated:
            env.render()
            print("Finished episode", episode, "with reward", total_reward)
            break

env.close()