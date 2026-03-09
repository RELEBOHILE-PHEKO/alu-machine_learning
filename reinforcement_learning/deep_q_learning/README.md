# Deep Q-Learning - Atari Breakout

## Description

This project implements a **Deep Q-Network (DQN)** to play Atari's **Breakout** using raw pixel input. The agent uses a convolutional neural network to approximate the Q-function, trained end-to-end directly from game frames using `keras`, `keras-rl`, and `gym`.

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Trains the DQN agent and saves the policy network to `policy.h5` |
| `play.py` | Loads `policy.h5` and displays the agent playing Breakout |

---

## Architecture

- Input: 4 stacked grayscale 84×84 frames
- 3 convolutional layers (32, 64, 64 filters)
- 1 fully connected layer (512 units)
- Output layer with one node per action

## Training Details

| Parameter | Value |
|-----------|-------|
| Memory replay buffer | 1,000,000 transitions |
| Warm-up steps | 50,000 |
| Total training steps | 1,750,000 |
| Target network update | every 10,000 steps |
| Optimizer | Adam (lr=0.00025) |
| Discount factor (γ) | 0.99 |
| Reward clipping | [−1, 1] |
| Policy (training) | `EpsGreedyQPolicy` |
| Policy (playing) | `GreedyQPolicy` |

---

## Requirements

- Python 3.5
- `keras`
- `keras-rl`
- `gym[atari]`

## Installation

```bash
pip install --user keras keras-rl gym[atari]
```

## Usage

```bash
# Train the agent (saves policy.h5)
python3 train.py

# Watch the agent play
python3 play.py
```

---
