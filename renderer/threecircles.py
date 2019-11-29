import pygame
import random
import os
import tqdm

from renderer.constants import TARGET_FPS, BACKGROUND

from renderer.world import World
from renderer.scene import Scene
from renderer.circle import Circle
from renderer.wall import Wall

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

MAX_VELOCITY = 150


class ThreeCircles(Scene):
    def __init__(self, headless=True, width=256, height=256, radius=20, bkg_color=BACKGROUND):
        circles = [
            Circle(
                position=(
                    random.randint(radius, width - radius),
                    random.randint(radius, height - radius),
                ),
                velocity=(
                    random.randint(-MAX_VELOCITY, MAX_VELOCITY),
                    random.randint(-MAX_VELOCITY, MAX_VELOCITY),
                ),
                r=radius,
            )
            for _ in range(3)
        ]

        super(ThreeCircles, self).__init__(
            "ThreeCircles", circles, COLORS,
            headless=headless, width=width, height=height, bkg_color=bkg_color
        )


def collect_data(
    sequence_length=500,
    num_sequences=400,
    data_directory="data/threecircles_128/train",
    seed=1337,
):
    random.seed(seed)

    for sequence in tqdm.tqdm(range(num_sequences)):
        scene = ThreeCircles(headless=True, width=64, height=64, radius=5)

        os.mkdir(os.path.join(data_directory, str(sequence)))

        for frame in range(sequence_length):
            scene.step()
            scene.draw()

            pygame.image.save(
                scene.binary_surface,
                os.path.join(data_directory, str(sequence), "%i.png" % (frame)),
            )


if __name__ == "__main__":
    # Training
    collect_data(
        num_sequences=400, seed=1337, data_directory="data/threecircles_64/train"
    )

    # Validation
    collect_data(
        num_sequences=20, seed=12398, data_directory="data/threecircles_64/val"
    )
