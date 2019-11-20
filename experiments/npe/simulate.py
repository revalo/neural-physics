"""Simulates a trained model.
"""

import numpy as np
from renderer.threecircles import ThreeCircles
from experiments.npe.datagen import normalize_position


def get_circle_state(scene, circle):
    return np.array(normalize_position(circle.position, scene.width, scene.height))


def get_input(state, past_steps):
    return np.concatenate(state[-past_steps:])


def show_simulation(model, width=256, height=256, radius=30, length=500, past_steps=2):
    scene = ThreeCircles(headless=False, width=width, height=height, radius=radius)

    # Buffer steps
    scene.step()
    scene.step()

    states = [[] for _ in scene.circles]

    for _ in range(10):
        scene.step()

        for i, circle in enumerate(scene.circles):
            states[i].append(get_circle_state(scene, circle))

    for frame in range(length):
        current_inputs = [get_input(state, past_steps) for state in states]

        for i, key_circle in enumerate(scene.circles):
            key_input = current_inputs[i]
            context_inputs = [
                current_inputs[i]
                for i, circle in enumerate(scene.circles)
                if circle != key_circle
            ]

            delta = model.predict(
                [
                    np.array([key_input]),
                    np.array([context_inputs[0]]),
                    np.array([context_inputs[1]]),
                    np.array([[1.0]]),
                    np.array([[1.0]]),
                ]
            )[0]

            states[i].append(states[i][-1] + delta / 10.0)

        for i, circle in enumerate(scene.circles):
            circle.position = (states[i][-1][0] * width, states[i][-1][1] * height)

        scene.draw()
