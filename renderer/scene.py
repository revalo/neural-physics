import pygame
import random
import os
import tqdm
import abc

from renderer.constants import TARGET_FPS, BACKGROUND

from renderer.world import World


class Scene:
    def __init__(self, scene_name, objects, colors, headless=True,
                 width=256, height=256, gravity=(0, 0), wall_elasticity=0.8,
                 bkg_color=BACKGROUND):
        assert(len(objects) == len(colors))

        self.headless = headless
        self.world = World(width, height, gravity, wall_elasticity)
        self.width = width
        self.height = height
        self.background_color = bkg_color

        self.objects = objects
        self.colors = colors

        for obj in self.objects:
            self.world.add_entity(obj)

        pygame.init()
        pygame.display.set_caption(scene_name)

        if not self.headless:
            self.screen = pygame.display.set_mode((width, height))

        self.compose_surface = pygame.Surface((width, height))
        self.binary_surface = pygame.Surface((width, height))
        self.object_surfaces = [
            pygame.Surface((width, height)) for obj in self.objects
        ]

        self.clock = pygame.time.Clock()

    def step(self):
        self.world.step(1.0 / TARGET_FPS)

    def draw(self):
        self.compose_surface.fill(self.background_color)
        self.binary_surface.fill((0, 0, 0))

        for i, obj in enumerate(self.objects):
            obj.draw(self.compose_surface, self.colors[i])
            obj.draw(self.binary_surface, (255, 255, 255))

            # Generate binary masked obj image.
            self.object_surfaces[i].fill((0, 0, 0))
            obj.draw(self.object_surfaces[i], (255, 255, 255))

        if not self.headless:
            self.screen.blit(self.compose_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def save_image(self, filename):
        pygame.image.save(self.compose_surface, filename)
