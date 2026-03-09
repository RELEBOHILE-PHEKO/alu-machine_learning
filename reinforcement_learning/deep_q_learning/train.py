#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout using keras-rl."""
import numpy as np
import gym
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

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
    """Build the CNN model for the DQN agent.

    Args:
        n_actions: Number of possible actions in the environment.

    Returns:
        Compiled Keras Sequential model.
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
    """Train DQN agent on Breakout and save policy to policy.h5."""
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(42)
    env.seed(42)

    n_actions = env.action_space.n
    model = build_model(n_actions)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = EpsGreedyQPolicy()
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

    callbacks = [
        ModelIntervalCheckpoint('dqn_breakout_weights_{step}.h5',
                                interval=250000)
    ]

    agent.fit(
        env,
        nb_steps=1750000,
        callbacks=callbacks,
        log_interval=10000,
        visualize=False
    )

    agent.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__ == '__main__':
    main()
