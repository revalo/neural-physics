"""Simulates a trained model.
"""

import math
import numpy as np
from renderer.rectangles import Rectangles
from renderer.threecircles import ThreeCircles
from experiments.npe_bt.datagen import normalize_state
from renderer.constants import TARGET_FPS, MAX_ANGULAR_VELOCITY, MAX_VELOCITY


def get_obj_state(scene, obj):
    return np.array(
        normalize_state(
            obj.position,
            obj.velocity,
            obj.shape.body.angle,
            obj.shape.body.angular_velocity,
            scene.width,
            scene.height,
        )
    )


def get_input(state, past_steps, scene):
    rv = np.concatenate(state[-past_steps:])

    rv[0] /= scene.width
    rv[1] /= scene.height
    rv[2] /= MAX_VELOCITY
    rv[3] /= MAX_VELOCITY
    rv[4] /= np.pi
    rv[5] /= MAX_ANGULAR_VELOCITY

    rv[6] /= scene.width
    rv[7] /= scene.height
    rv[8] /= MAX_VELOCITY
    rv[9] /= MAX_VELOCITY
    rv[10] /= np.pi
    rv[11] /= MAX_ANGULAR_VELOCITY

    return rv


def show_simulation(model, width=256, height=256, radius=30, length=500, past_steps=2):
    # TODO(ayue): Enable scene picking.
    scene = Rectangles(headless=False, rand_height=False)
    # scene = ThreeCircles(headless=False, width=width, height=height, radius=radius)

    states = [[] for _ in scene.objects]

    for _ in range(3):
        scene.step()

        for i, obj in enumerate(scene.objects):
            states[i].append(
                np.array(
                    [
                        obj.position[0],
                        obj.position[1],
                        obj.velocity[0],
                        obj.velocity[1],
                        obj.shape.body.angle,
                        obj.shape.body.angular_velocity,
                    ]
                )
            )

    for frame in range(length):
        current_inputs = [get_input(state, past_steps, scene) for state in states]

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

            dv = delta[:2]
            da = delta[-1]

            # if abs(da) <= 1e-2:
            #     da = 0.0

            # if np.linalg.norm(dv) <= 1e-2:
            #     dv = np.zeros((2,))

            # Pixels / Second
            new_velocity = states[i][-1][2:4] + dv * MAX_VELOCITY
            new_angular_velocity = states[i][-1][5] + da * MAX_ANGULAR_VELOCITY

            # Pixels
            new_position = states[i][-1][:2] + new_velocity / TARGET_FPS
            new_angle = states[i][-1][4] + new_angular_velocity / TARGET_FPS

            if new_angle > np.pi:
                new_angle += -2 * np.pi
            if new_angle < -np.pi:
                new_angle += 2 * np.pi

            new_state = np.hstack(
                (new_position, new_velocity, [new_angle], [new_angular_velocity])
            )

            states[i].append(new_state)

        for i, circle in enumerate(scene.objects):
            p = (states[i][-1][0], states[i][-1][1])
            circle.position = p
            circle.shape.body.angle = states[i][-1][4]

        scene.draw()
