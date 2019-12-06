import pygame
import random
import os
import tqdm

from renderer.constants import TARGET_FPS

from renderer.world import World
from renderer.circle import Circle
from renderer.wall import Wall

BACKGROUND = (255, 255, 255)

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

MAX_VELOCITY = 150


class ThreeCircles(object):
    def __init__(self, headless=True, width=256, height=256, radius=20):
        self.headless = headless
        self.world = World(width, height)
        self.width = width
        self.height = height
        self.radius = radius

        self.circles = [
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

        for circle in self.circles:
            self.world.add_entity(circle)

        pygame.init()
        pygame.display.set_caption("ThreeCircles")

        if not self.headless:
            self.screen = pygame.display.set_mode((width, height))

        self.compose_surface = pygame.Surface((width, height))
        self.binary_surface = pygame.Surface((width, height))
        self.circle_surfaces = [
            pygame.Surface((width, height)) for circle in self.circles
        ]

        self.clock = pygame.time.Clock()

    def step(self):
        self.world.step(1.0 / TARGET_FPS)

    def draw(self):
        self.compose_surface.fill(BACKGROUND)
        self.binary_surface.fill((0, 0, 0))

        for i, circle in enumerate(self.circles):
            circle.draw(self.compose_surface, COLORS[i])
            circle.draw(self.binary_surface, (255, 255, 255))

            # Generate binary masked circle image.
            self.circle_surfaces[i].fill((0, 0, 0))
            circle.draw(self.circle_surfaces[i], (255, 255, 255))

        if not self.headless:
            self.screen.blit(self.compose_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def save_image(self, filename):
        pygame.image.save(self.compose_surface, filename)


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
