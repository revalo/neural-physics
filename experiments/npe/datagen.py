"""Handles using the scene simulations to generate NPE training data.
"""

import tqdm
import random
import renderer.threecircles as threecircles
import numpy as np

import pygame

from renderer.constants import TARGET_FPS, MAX_VELOCITY


def normalize_position(position, width, height):
    x, y = position

    return (
        x / width,
        y / height,
    )


def normalize_velocity(velocity):
    vx, vy = velocity

    return (
        vx / MAX_VELOCITY,
        vy / MAX_VELOCITY,
    )


def get_complexity(circle_circle, circle_wall):
    """Given number of circle-circle collisions and number of circle-wall collisions,
    find a number that tells you how complex this training example is.
    """

    return circle_wall + 3 * circle_circle


def collect_data(
    num_sequences=10000,
    seed=1337,
    sequence_length=500,
    history=2,
    max_pairs=2,
    width=256,
    height=256,
    radius=30,
    neighborhood_mask=60 * 3,
):
    pygame.init()
    pygame.quit()
    random.seed(seed)

    X = []
    y = []
    complexities = []

    for sequence in tqdm.tqdm(range(num_sequences)):
        # TODO(ayue): Enable scene picking.
        scene = threecircles.ThreeCircles(
            headless=True, width=width, height=height, radius=radius
        )

        key_circle = random.choice(scene.circles)
        context_circles = [circle for circle in scene.circles if circle != key_circle]

        assert len(context_circles) < len(scene.objects)

        for frame in range(sequence_length):
            # Calculate setup.
            key_state = []
            context_states = [[] for _ in range(max_pairs)]

            used = []

            scene.reset_collision_counters()
            for step in range(history):
                scene.step()

                key_state.extend(normalize_position(key_circle.position, width, height))
                key_state.extend(normalize_velocity(key_circle.velocity))

                for i, circle in enumerate(context_circles):
                    context_states[i].extend(
                        normalize_position(circle.position, width, height)
                    )
                    context_states[i].extend(normalize_velocity(circle.velocity))
                    if True or (
                        np.linalg.norm(
                            np.array(circle.position) - np.array(key_circle.position)
                        )
                        <= neighborhood_mask
                    ):
                        if step == history - 1:
                            used.append(np.array([1.0]))
                    else:
                        if step == 0:
                            used.append(np.array([0.0]))

            key_state = np.array(key_state)
            for i in range(len(context_states)):
                context_states[i] = np.array(context_states[i])

            # Calculate answer.
            prev_velocity = np.array(normalize_velocity(key_circle.velocity))
            scene.step()
            next_velocity = np.array(normalize_velocity(key_circle.velocity))

            # Final state is the change in velocity.
            final_state = next_velocity - prev_velocity

            X.append([key_state] + context_states + used)
            y.append(final_state)
            complexities.append(get_complexity(scene.circle_circle, scene.circle_wall))

    return X, y, complexities


if __name__ == "__main__":
    X, y, complexities = collect_data()
