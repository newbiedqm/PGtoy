# PGtoy

Policy Gradient (PG) methods on toy examples using Gymnasium.

## Overview

This repository contains implementations of Policy Gradient algorithms for reinforcement learning, demonstrated on simple Gymnasium environments. The current implementation includes:

- **REINFORCE**: Monte Carlo Policy Gradient algorithm

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Example: CartPole-v1

Run the CartPole example:

```bash
python example_cartpole.py
```

### Training on Different Environments

Train REINFORCE on any Gymnasium environment with discrete action space:

```bash
python train.py --env CartPole-v1 --episodes 1000 --lr 0.01
```

Available arguments:
- `--env`: Gymnasium environment name (default: CartPole-v1)
- `--episodes`: Number of training episodes (default: 1000)
- `--lr`: Learning rate (default: 1e-3)
- `--gamma`: Discount factor (default: 0.99)
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--test-episodes`: Number of test episodes (default: 10)
- `--render`: Render environment during testing

### Example Environments

Try different toy environments:
- `CartPole-v1`: Balance a pole on a cart
- `Acrobot-v1`: Swing up a two-link robot
- `MountainCar-v0`: Drive a car up a mountain (requires modification for continuous rewards)

## Algorithm: REINFORCE

REINFORCE is a Monte Carlo policy gradient algorithm that:
1. Collects full episode trajectories
2. Calculates discounted returns
3. Updates policy to increase probability of actions with higher returns

The implementation uses:
- Policy network with one hidden layer
- Softmax policy for discrete action spaces
- Adam optimizer
- Baseline subtraction (normalized returns) to reduce variance

## Project Structure

```
PGtoy/
├── reinforce.py          # REINFORCE algorithm implementation
├── train.py              # Training script with command-line arguments
├── example_cartpole.py   # Quick example on CartPole
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.7+
- gymnasium >= 0.29.0
- numpy >= 1.24.0
- torch >= 2.0.0
- matplotlib >= 3.7.0
