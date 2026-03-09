#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent loaded from policy.h5."""
import numpy as np
import gym
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    """Processor for Atari game frames."""

    def process_observation(self, observation):
        """Process a raw Atari frame to grayscale 84x84.

        Args:
            observation: Raw RGB frame from the environment.

        Returns:
            Processed grayscale uint8 image of shape (84, 84).
        """
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE, Image.ANTIALIAS).convert('L')
        return np.array(img, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize pixel values to [0, 1].

        Args:
            batch: Batch of states.

        Returns:
            Float32 normalized batch.
        """
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip reward to [-1, 1].

        Args:
            reward: Raw reward from the environment.

        Returns:
            Clipped reward.
        """
        return np.clip(reward, -1.0, 1.0)


def build_model(n_actions):
    """Build the CNN model matching the architecture used in training.

    Args:
        n_actions: Number of possible actions in the environment.

    Returns:
        Keras Sequential model (uncompiled weights loaded separately).
    """
    model = Sequential([
        Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE),
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(n_actions, activation='linear'),
    ])
    return model


def main():
    """Load trained policy from policy.h5 and display one game episode."""
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(42)
    env.seed(42)

    n_actions = env.action_space.n
    model = build_model(n_actions)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    agent = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(Adam(lr=0.00025), metrics=['mae'])
    agent.load_weights('policy.h5')

    agent.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == '__main__':
    main()
