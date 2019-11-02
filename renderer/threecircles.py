import pygame
import random
import os

from constants import TARGET_FPS

from world import World
from circle import Circle
from wall import Wall

BACKGROUND = (255, 255, 255)

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

MAX_VELOCITY = 25


class ThreeCircles(object):
    def __init__(self, headless=True, width=256, height=256, radius=20):
        self.headless = headless
        self.world = World(width, height)

        self.circles = [
            Circle(
                position=(
                    random.randint(radius, width - radius),
                    random.randint(radius, height - radius),
                ),
                velocity=(
                    random.randrange(-MAX_VELOCITY, MAX_VELOCITY),
                    random.randrange(-MAX_VELOCITY, MAX_VELOCITY),
                ),
                r=radius,
            )
            for _ in range(3)
        ]

        for circle in self.circles:
            self.world.add_entity(circle)

        pygame.init()
        pygame.display.set_caption('ThreeCircles')

        if self.headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.screen = pygame.display.set_mode((width, height))

        self.compose_surface = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()

    def step(self):
        self.world.step(1.0 / TARGET_FPS)


    def draw(self):
        self.compose_surface.fill(BACKGROUND)

        for i, circle in enumerate(self.circles):
            circle.draw(self.compose_surface, COLORS[i])

        if not self.headless:
            self.screen.blit(self.compose_surface, (0, 0))

        pygame.display.flip()

        if not self.headless:
            self.clock.tick(TARGET_FPS)

    def save_image(self, filename):
        pygame.image.save(self.compose_surface, filename)


if __name__ == "__main__":
    scene = ThreeCircles(headless=True)

    for i in range(1000):
        scene.step()
        scene.draw()
        scene.save_image("%i.png" % i)
