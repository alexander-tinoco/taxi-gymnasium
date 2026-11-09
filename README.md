# taxi-gymnasium

Tabular Q-learning agent that solves Gymnasium's `Taxi-v4` environment. It trains a Q-table, saves it to disk, and replays the learned policy in a pygame window.

## Project status

Small learning project, finished. The committed Q-table solves the task: over 200 evaluation episodes it delivered the passenger every time, with a mean reward of about 7.8.

## The problem

In `Taxi-v4` a taxi moves on a 5×5 grid with four pickup/drop-off locations. It must go to the passenger, pick them up, drive to their destination and drop them off.

| | |
|---|---|
| **States** | 500 (taxi position × passenger location × destination) |
| **Actions** | 6: south, north, east, west, pickup, drop-off |
| **Rewards** | −1 per step, +20 for a correct drop-off, −10 for an illegal pickup or drop-off |

Because the state and action spaces are small and discrete, the value function fits in a plain table of 500 × 6 numbers. No neural network is needed.

## How it learns

Standard epsilon-greedy Q-learning. After each step the table is updated with:

```
Q[s, a] = (1 − α) · Q[s, a] + α · (r + γ · max Q[s', :])
```

| Parameter | Value | Meaning |
|---|---|---|
| `ALPHA` | 0.9 | Learning rate |
| `GAMMA` | 0.95 | Discount factor |
| `EPSILON_START` → `MIN_EPSILON` | 1.0 → 0.01 | Exploration rate, multiplied by `EPSILON_DECAY` (0.995) after every episode |
| `NUM_EPISODES` | 10,000 | Training episodes |
| `MAX_STEPS` | 100 | Step limit per episode |

A checkpoint of the table is saved every 1,000 episodes, so an interrupted run does not lose its progress. Training takes a few seconds on a laptop.

## Repository layout

```
src/taxi/
├── agent.py            # training loop, policy replay and the entry point
├── __init__.py         # exposes main() for the `taxi` command
└── q_table_taxi.npy    # the trained Q-table (500 × 6, float64)
```

## Running it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/alexander-tinoco/taxi-gymnasium.git
cd taxi-gymnasium
uv run taxi
```

The first run installs the dependencies (`gymnasium`, `pygame`) and then:

1. **Loads** `src/taxi/q_table_taxi.npy` if it exists and skips training.
2. **Otherwise trains** a new table and saves it there.
3. **Opens a pygame window** and plays 5 episodes with the greedy policy, printing the total reward of each.

To retrain from scratch, delete the saved table and run it again:

```bash
rm src/taxi/q_table_taxi.npy
uv run taxi
```

Without a display (for example over SSH), set `SDL_VIDEODRIVER=dummy` to run it headless.

## Known limitations

- **Tabular only.** It works because the state space has 500 entries. A larger environment would need function approximation, such as a DQN.
- **The demo is not a benchmark.** It replays 5 episodes. The success figure above comes from a separate 200-episode run, not from a test in the repository.
- **Fixed hyperparameters.** They are constants at the top of `agent.py`, with no command-line options.
- **No automated tests.**

## License

MIT. See [LICENSE](LICENSE).
