"""Implements an environment that takes in an initial configuration and spits out
rewards.
"""

import tensorflow as tf
import numpy as np

from renderer.constants import TARGET_FPS
from renderer.threecircles import ThreeCircles, MAX_VELOCITY
from experiments.npe.simulate import get_circle_state


def configuration_to_position(x, y, width, height, radius):
    # Scale the position.
    x, y = x * width, y * height

    # Clamp to valid position.
    return min(max(x, radius), width - radius), min(max(y, radius), height - radius)


def configuration_to_velocity(vx, vy):
    # Scale the velocity.
    return (vx * 2.0 - 1.0) * MAX_VELOCITY, (vy * 2.0 - 1.0) * MAX_VELOCITY


class Environment(object):
    def __init__(
        self,
        model,
        episodes=100,
        batch_size=128,
        width=256,
        height=256,
        radius=30,
        past_timesteps=2,
    ):
        """Initializes the environment.
        """

        self.model = model
        self.episodes = episodes
        self.batch_size = batch_size
        self.past_timesteps = past_timesteps

        self.width = width
        self.height = height
        self.radius = radius

        # Setup scene replicas.
        self.scenes = [
            ThreeCircles(width=self.width, height=self.height, radius=self.radius)
            for _ in range(self.batch_size)
        ]

    def get_reward(self, configurations):
        """Returns the reward given configuration vectors.

        The configuration vector specifies the initial locations and velocities of the
        three balls. [x1, y1, vx1, vy1, ...]

        This function is passed in multiple configurations as a batch, shape of
        configurations should be (batch_size, 4*3).
        """

        num_circles = len(self.scenes[0].circles)

        # Initialize scenes with initial velocities and positions from configs.
        print("Scene Setup.")
        for s_i, scene in enumerate(self.scenes):
            configuration = configurations[s_i]

            for c_i, circle in enumerate(scene.circles):
                x, y, vx, vy = configuration[c_i : c_i + 4]

                circle.position = configuration_to_position(
                    x, y, self.width, self.height, self.radius
                )
                circle.velocity = configuration_to_velocity(vx, vy)

            # Buffer step to correct for collisions.
            # TODO(shreyask): Maybe this is not needed and might harm the model.
            scene.step()

        # Generate trajectories.

        # There are two ways to approach this, either the model goes completely on its
        # own for all episodes, or it's hand held and make singular predictions from
        # given true values. Right now we're doing the second approach.

        actual_states = np.zeros(
            (len(self.scenes), num_circles, self.episodes, 2), dtype=np.float32
        )

        predicted_states = np.zeros(
            (len(self.scenes), num_circles, self.episodes, 2), dtype=np.float32
        )

        errors = []

        # Run simulations.
        print("Generate simulations.")
        for episode in range(self.episodes):
            for s_i, scene in enumerate(self.scenes):
                scene.step()

                for c_i, circle in enumerate(scene.circles):
                    actual_states[s_i, c_i, episode] = get_circle_state(scene, circle)

        # Make model predictions.
        print("Make predictions.")
        for episode in range(self.past_timesteps, self.episodes):
            current_state = actual_states[
                :, :, episode - self.past_timesteps : episode, :
            ].reshape((len(self.scenes), num_circles, self.past_timesteps * 2))

            for focus_circle_index in range(num_circles):
                context_indices = [
                    i for i in range(num_circles) if i != focus_circle_index
                ]

                predictions = self.model.predict(
                    # Focus states.
                    [current_state[:, focus_circle_index]]
                    # Context states.
                    + [
                        current_state[:, context_index]
                        for context_index in context_indices
                    ]
                    # Used masks.
                    + [
                        np.ones((len(self.scenes), 1), dtype=np.float32)
                        for _ in context_indices
                    ]
                )

                predicted_states[:, focus_circle_index, episode] = (
                    actual_states[:, focus_circle_index, episode - 1]
                    + predictions / TARGET_FPS
                )

                mse = np.linalg.norm(
                    actual_states[:, focus_circle_index, episode]
                    - predicted_states[:, focus_circle_index, episode]
                )

                errors.append(mse)

        return np.mean(errors), actual_states, predicted_states
