import random
import os
import tqdm
import pymunk
import pygame

from renderer.constants import TARGET_FPS, BACKGROUND, MAX_VELOCITY

from renderer.world import World
from renderer.scene import Scene
from renderer.circle import Circle
from renderer.rect import Rect
from renderer.wall import Wall

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
]


class ThreeCircles(object):
    def __init__(self, headless=True, width=256, height=256, radius=30):
        self.headless = headless
        self.world = World(width, height, wall_elasticity=1.0, wall_friction=0.0)
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
            # Rect(
            #     position=(
            #         random.randint(radius, width - radius),
            #         random.randint(radius, height - radius),
            #     ),
            #     velocity=(
            #         random.randint(-MAX_VELOCITY, MAX_VELOCITY),
            #         random.randint(-MAX_VELOCITY, MAX_VELOCITY),
            #     ),
            #     width=int(radius * 2),
            #     height=int(radius * 2),
            # )
            for _ in range(3)
        ]

        self.objects = []

        for circle in self.circles:
            self.world.add_entity(circle)
            self.objects.append(circle)

        # Collision handler.
        self.collision_handler = self.world.space.add_default_collision_handler()
        self.collision_handler.begin = self.handle_collison

        self.circle_circle = 0
        self.circle_wall = 0

        if not self.headless:

            pygame.init()
            pygame.display.set_caption("ThreeCircles")
            self.screen = pygame.display.set_mode((width, height))

            self.compose_surface = pygame.Surface((width, height))
            self.binary_surface = pygame.Surface((width, height))
            self.circle_surfaces = [
                pygame.Surface((width, height)) for circle in self.circles
            ]
            self.clock = pygame.time.Clock()

    def handle_collison(self, arbiter, space, data):
        circles = len(
            [s for s in arbiter.shapes if isinstance(s, pymunk.shapes.Circle)]
        )
        walls = len([s for s in arbiter.shapes if isinstance(s, pymunk.shapes.Poly)])

        if circles == 2:
            self.circle_circle += 1

        if circles == 1 and walls == 1:
            self.circle_wall += 1

        return True

    def reset_collision_counters(self):
        self.circle_circle = 0
        self.circle_wall = 0

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

    def draw_trails(self):
        for i, circle in enumerate(self.circles):
            circle.draw_trail(self.compose_surface)

        if not self.headless:
            self.screen.blit(self.compose_surface, (0, 0))
            pygame.display.flip()

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
