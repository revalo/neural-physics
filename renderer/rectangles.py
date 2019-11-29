import pygame
import random
import os
import tqdm

from renderer.constants import TARGET_FPS, BACKGROUND

from renderer.world import World
from renderer.scene import Scene
from renderer.rect import Rect
from renderer.wall import Wall

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (0, 0, 0),
]

MAX_VELOCITY = 150

NUM_RECTANGLES = 6


class Rectangles(Scene):
    # position is the center of the box
    def __init__(self, headless=True, width=256, height=256,
                 rect_height=20, rect_width=20, bkg_color=BACKGROUND):
        rects = [
            Rect(
                position=(
                    random.randint(rect_width, width - rect_width),
                    random.randint(rect_height, height - rect_height),
                ),
                velocity=(
                    random.randint(0,0),
                    random.randint(0,0),
                ),
                height=rect_height,
                width=rect_width,
            )
            for _ in range(NUM_RECTANGLES)
        ]

        super(Rectangles, self).__init__(
            "SixRectangles", rects, COLORS, headless=headless,
            width=width, height=height, gravity=(0, 9.8), wall_elasticity=0.8,
            bkg_color=bkg_color
        )


def collect_data(
    sequence_length=500,
    num_sequences=400,
    data_directory="data/rectangles_128/train",
    seed=1337,
):
    random.seed(seed)

    for sequence in tqdm.tqdm(range(num_sequences)):
        scene = Rectangles(headless=True, width=64, height=64, rect_height=5, rect_width=5)

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
        num_sequences=400, seed=1337, data_directory="data/rectangles_64/train"
    )

    # Validation
    collect_data(
        num_sequences=20, seed=12398, data_directory="data/rectangles_64/val"
    )
