"""Handles using the scene simulations to generate NPE training data.
"""

import tqdm
import random
import numpy as np
import math

from renderer.constants import TARGET_FPS
import renderer.rectangles as rectangles

def normalize_position(position, angle, width, height):
    x, y = position

    return (
        x / width,
        y / height,
        angle / (2 * math.pi), #maybe it's pi?
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
        scene = rectangles.Rectangles(
            headless=True, rand_height=False
        )

        # TODO(shreyask): Think about velocity Box2D multiplier.
        key_circle = random.choice(scene.objects)
        context_circles = [circle for circle in scene.objects if circle != key_circle]

        assert len(context_circles) < len(scene.objects)

        for frame in range(sequence_length):
            # Calculate setup.
            key_state = []
            context_states = [[] for _ in range(max_pairs)]

            used = [np.array([1.0]) for _ in range(len(context_circles))] + [
                np.array([0.0]) for _ in range(max_pairs - len(context_circles))
            ]

            for step in range(history):
                scene.step()

                key_state.extend(normalize_position(key_circle.position, key_circle.shape.body.angle, width, height))

                for i, circle in enumerate(context_circles):
                    context_states[i].extend(
                        normalize_position(circle.position, circle.shape.body.angle, width, height)
                    )

                for i in range(len(context_circles), max_pairs):
                    context_states[i].extend([0.0, 0.0])

            key_state = np.array(key_state)
            for i in range(len(context_states)):
                context_states[i] = np.array(context_states[i])

            # Calculate answer.
            prev_position = normalize_position(key_circle.position, key_circle.shape.body.angle, width, height)
            scene.step()
            next_position = normalize_position(key_circle.position, key_circle.shape.body.angle, width, height)

            # Final state is the velocity.
            final_state = TARGET_FPS * (
                np.array(next_position) - np.array(prev_position)
            )

            X.append([key_state] + context_states + used)
            y.append(final_state)

    return X, y


if __name__ == "__main__":
    X, y = collect_data()
