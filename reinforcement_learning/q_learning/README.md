# Q-Learning - Reinforcement Learning

## Description

This project implements Q-learning, a model-free reinforcement learning algorithm, applied to OpenAI Gym's FrozenLake environment. The agent learns an optimal policy to navigate a frozen lake grid from start (S) to goal (G) while avoiding holes (H).

## Environment

**FrozenLake-v0**: A grid-world where the agent must traverse a frozen lake. The map contains:
- `S` — Starting position
- `F` — Frozen surface (safe)
- `H` — Hole (episode ends, negative reward)
- `G` — Goal (episode ends, positive reward)

## Concepts

### Markov Decision Process (MDP)
A mathematical framework for modeling decision-making where outcomes are partly random and partly controlled by the agent. Defined by states, actions, transition probabilities, and rewards.

### Q-Learning
A model-free, off-policy temporal difference learning algorithm that learns the value of an action in a given state. Updates are done using the Bellman equation:

```
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `α` (alpha) — learning rate
- `γ` (gamma) — discount factor
- `r` — reward
- `s'` — next state

### Epsilon-Greedy
A strategy balancing **exploration** (random actions) and **exploitation** (best known action). With probability `ε`, the agent explores; otherwise it exploits the Q-table.

## Files

| File | Description |
|------|-------------|
| `0-load_env.py` | Loads the FrozenLake environment from OpenAI Gym |
| `1-q_init.py` | Initializes the Q-table as a zero matrix |
| `2-epsilon_greedy.py` | Implements epsilon-greedy action selection |
| `3-q_learning.py` | Trains the agent using the Q-learning algorithm |
| `4-play.py` | Runs a trained agent through one episode |

## Requirements

- Python 3.5 (Ubuntu 16.04 LTS)
- numpy 1.15
- gym 0.7

## Installation

```bash
pip install --user gym
```

## Usage

```bash
# Load environment
./0-load_env.py

# Initialize Q-table
./1-q_init.py

# Train the agent
python3 3-main.py

# Play an episode with trained agent
python3 4-main.py
```
