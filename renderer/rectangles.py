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


class Rectangles(object):
    def __init__(
        self, headless=True, width=256, height=256, rect_height=20, rect_width=20
    ):
        self.headless = headless
        self.world = World(width, height, gravity=(0,9.8))
        self.width = width
        self.height = height

        self.rects = [
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
            for _ in range(6)
        ]

        for r in self.rects:
            self.world.add_entity(r)

        pygame.init()
        pygame.display.set_caption("SixRectangles")

        if not self.headless:
            self.screen = pygame.display.set_mode((width, height))

        self.compose_surface = pygame.Surface((width, height))
        self.binary_surface = pygame.Surface((width, height))
        self.rect_surfaces = [pygame.Surface((width, height)) for r in self.rects]

        self.clock = pygame.time.Clock()

    def step(self):
        self.world.step(1.0 / TARGET_FPS)

    def draw(self):
        self.compose_surface.fill(BACKGROUND)
        self.binary_surface.fill((0, 0, 0))

        for i, r in enumerate(self.rects):
            r.draw(self.compose_surface, COLORS[i])
            r.draw(self.binary_surface, (255, 255, 255))

            # Generate binary masked circle image.
            self.rect_surfaces[i].fill((0, 0, 0))
            r.draw(self.rect_surfaces[i], (255, 255, 255))

        if not self.headless:
            self.screen.blit(self.compose_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def save_image(self, filename):
        pygame.image.save(self.compose_surface, filename)


def collect_data(
    sequence_length=500,
    num_sequences=400,
    data_directory="data/sixrectangles_128/train",
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
