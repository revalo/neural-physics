"""Handles using the scene simulations to generate NPE training data.
"""

import tqdm
import random
import numpy as np
import math

from renderer.constants import TARGET_FPS, MAX_VELOCITY, MAX_ANGULAR_VELOCITY
from renderer.rectangles import Rectangles
from renderer.threecircles import ThreeCircles


def normalize_state(position, velocity, angle, angular_velocity, width, height):
    x, y = position
    vx, vy = velocity

    return (
        x / width,
        y / height,
        vx / MAX_VELOCITY,
        vy / MAX_VELOCITY,
        angle / np.pi,  # maybe it's pi?
        angular_velocity / MAX_ANGULAR_VELOCITY,
    )


def collect_data(
    num_sequences=10000,
    seed=1337,
    sequence_length=500,
    history=2,
    max_pairs=5,
    width=256,
    height=256,
    radius=30,
):
    random.seed(seed)

    X = []
    y = []

    for sequence in tqdm.tqdm(range(num_sequences)):
        # TODO(ayue): Enable scene picking.
        scene = Rectangles(headless=True, rand_height=False, wall_elasticity=1.0)
        # scene = ThreeCircles(headless=True, width=width, height=height, radius=radius)

        # TODO(shreyask): Think about velocity Box2D multiplier.
        key_object = random.choice(scene.objects)
        context_objects = [obj for obj in scene.objects if obj != key_object]

        assert len(context_objects) < len(scene.objects)

        for frame in range(sequence_length):
            # Calculate setup.
            key_state = []
            context_states = [[] for _ in range(len(scene.objects) - 1)]

            used = []

            for step in range(history):
                scene.step()

                key_state.extend(
                    normalize_state(
                        key_object.position,
                        key_object.velocity,
                        key_object.shape.body.angle,
                        key_object.shape.body.angular_velocity,
                        width,
                        height,
                    )
                )

                for i, obj in enumerate(context_objects):
                    context_states[i].extend(
                        normalize_state(
                            obj.position,
                            obj.velocity,
                            obj.shape.body.angle,
                            obj.shape.body.angular_velocity,
                            width,
                            height,
                        )
                    )

                    if step == history - 1:
                        used.append(np.array([1.0]))

            key_state = np.array(key_state)
            for i in range(len(context_states)):
                context_states[i] = np.array(context_states[i])

            # Calculate answer.
            _, _, p_vx, p_vy, _, p_av = normalize_state(
                key_object.position,
                key_object.velocity,
                key_object.shape.body.angle,
                key_object.shape.body.angular_velocity,
                width,
                height,
            )
            scene.step()
            _, _, n_vx, n_vy, _, n_av = normalize_state(
                key_object.position,
                key_object.velocity,
                key_object.shape.body.angle,
                key_object.shape.body.angular_velocity,
                width,
                height,
            )

            dv = np.array([n_vx, n_vy]) - np.array([p_vx, p_vy])
            dav = np.array([n_av]) - np.array([p_av])

            # Final state is the change in velocities.
            final_state = np.hstack((dv, dav))

            X.append([key_state] + context_states + used)
            y.append(final_state)

    # TODO(shreyask): Actually return complexities.
    return X, y, []


if __name__ == "__main__":
    X, y = collect_data(num_sequences=1, seed=1337)
    print(X[0])
    print(y[0])
