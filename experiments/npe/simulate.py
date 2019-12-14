"""Simulates a trained model.
"""

import numpy as np
from renderer.threecircles import ThreeCircles
from experiments.npe.datagen import normalize_position, normalize_velocity
from renderer.constants import TARGET_FPS, MAX_VELOCITY


def get_circle_state(scene, circle):
    return np.array(
        normalize_position(circle.position, scene.width, scene.height)
        + normalize_velocity(circle.velocity)
    )


def get_input(state, past_steps, scene):
    rv = np.concatenate(state[-past_steps:])

    rv[0] /= scene.width
    rv[1] /= scene.height
    rv[2] /= MAX_VELOCITY
    rv[3] /= MAX_VELOCITY

    rv[4] /= scene.width
    rv[5] /= scene.height
    rv[6] /= MAX_VELOCITY
    rv[7] /= MAX_VELOCITY

    return rv


import pygame
from pygame.locals import *


def get_mask(key_state, context_state, neighborhood_mask=60 * 3):
    return 1.0


def show_simulation(
    model,
    width=256,
    height=256,
    radius=30,
    length=500,
    past_steps=2,
    neighborhood_mask=60 * 3,
):
    scene = ThreeCircles(headless=False, width=width, height=height, radius=radius)

    # Buffer steps
    scene.step()
    scene.step()

    states = [[] for _ in scene.circles]

    for _ in range(2):
        scene.step()

        for i, circle in enumerate(scene.circles):
            states[i].append(
                np.array(
                    [
                        circle.position[0],
                        circle.position[1],
                        circle.velocity[0],
                        circle.velocity[1],
                    ]
                )
            )

    for frame in range(length):
        current_inputs = [get_input(state, past_steps, scene) for state in states]

        for i, key_circle in enumerate(scene.circles):
            key_input = current_inputs[i]
            context_inputs = [
                current_inputs[i]
                for i, circle in enumerate(scene.circles)
                if circle != key_circle
            ]

            inp = (
                [np.array([key_input]),]
                + [np.array([context_inputs[i]]) for i in range(len(scene.circles) - 1)]
                + [
                    np.array([[get_mask(key_circle.position, circle.position)]])
                    for circle in scene.circles
                    if circle != key_circle
                ]
            )

            dv = model(inp)[0]

            # Pixels / Second
            new_velocity = states[i][-1][2:] + dv * MAX_VELOCITY

            # Pixels
            new_position = states[i][-1][:2] + new_velocity / TARGET_FPS

            new_state = np.hstack((new_position, new_velocity))

            states[i].append(new_state)

        for i, circle in enumerate(scene.circles):
            circle.position = (states[i][-1][0], states[i][-1][1])

        scene.draw()


def show_simulation_variational(
    model, width=256, height=256, radius=30, length=500, past_steps=2
):
    scene = ThreeCircles(headless=False, width=width, height=height, radius=radius)

    # Buffer steps
    scene.step()
    scene.step()

    states_init = [[] for _ in scene.circles]

    for _ in range(10):
        scene.step()
        scene.draw()
        for i, circle in enumerate(scene.circles):
            states_init[i].append(
                np.array(
                    [
                        circle.position[0],
                        circle.position[1],
                        circle.velocity[0],
                        circle.velocity[1],
                    ]
                )
            )

    for simulations in range(50):
        states = [x[:] for x in states_init]

        for circle in scene.circles:
            circle.last_pos = None

        for frame in range(length):
            current_inputs = [get_input(state, past_steps, scene) for state in states]

            for i, key_circle in enumerate(scene.circles):
                key_input = current_inputs[i]
                context_inputs = [
                    current_inputs[i]
                    for i, circle in enumerate(scene.circles)
                    if circle != key_circle
                ]

                inp = (
                    [np.array([key_input]),]
                    + [
                        np.array([context_inputs[i]])
                        for i in range(len(scene.circles) - 1)
                    ]
                    + [
                        np.array([[get_mask(key_circle.position, circle.position)]])
                        for circle in scene.circles
                        if circle != key_circle
                    ]
                )

                dv = model(inp)[0]

                # Pixels / Second
                new_velocity = states[i][-1][2:] + dv * MAX_VELOCITY

                # Pixels
                new_position = states[i][-1][:2] + new_velocity / TARGET_FPS

                new_state = np.hstack((new_position, new_velocity))

                states[i].append(new_state)

            for i, circle in enumerate(scene.circles):
                circle.position = (states[i][-1][0], states[i][-1][1])

            scene.draw_trails()
