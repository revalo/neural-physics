"""Simulates a trained model.
"""

import math
import numpy as np
from renderer.rectangles import Rectangles
from experiments.npe_bt.datagen import normalize_state
from renderer.constants import TARGET_FPS


def get_circle_state(scene, circle):
    return np.array(
        normalize_state(
            circle.position, circle.shape.body.angle, scene.width, scene.height
        )
    )


def get_input(state, past_steps):
    return np.concatenate(state[-past_steps:])


def show_simulation(model, width=256, height=256, radius=30, length=500, past_steps=2):
    # TODO(ayue): Enable scene picking.
    scene = Rectangles(headless=False, rand_height=False)

    # Buffer steps
    scene.step()
    scene.step()

    states = [[] for _ in scene.objects]

    for _ in range(10):
        scene.step()

        for i, circle in enumerate(scene.objects):
            states[i].append(get_circle_state(scene, circle))

    for frame in range(length):
        current_inputs = [get_input(state, past_steps) for state in states]

        for i, key_circle in enumerate(scene.objects):
            key_input = current_inputs[i]
            context_inputs = [
                current_inputs[i]
                for i, circle in enumerate(scene.objects)
                if circle != key_circle
            ]

            delta = model.predict(
                [
                    np.array([key_input]),
                    np.array([context_inputs[0]]),
                    np.array([context_inputs[1]]),
                    np.array([context_inputs[2]]),
                    np.array([context_inputs[3]]),
                    np.array([context_inputs[4]]),
                    # Factor for the edge weights of the corresponding context object.
                    # 0 if no object, 1 if object with full importance.
                    np.array([[1.0]]),
                    np.array([[1.0]]),
                    np.array([[1.0]]),
                    np.array([[1.0]]),
                    np.array([[1.0]]),
                ]
            )[0]
            print(delta)
            states[i].append(states[i][-1] + delta / 1000.0)

        for i, circle in enumerate(scene.objects):
            circle.position = (states[i][-1][0] * width, states[i][-1][1] * height)
            circle.shape.body.angle = states[i][-1][2] * 2 * math.pi

        scene.draw()
