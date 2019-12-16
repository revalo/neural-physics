import pygame
import pymunk

from renderer.entity import Entity


class Circle(Entity):
    def __init__(self, position, velocity, r, mass=10):
        super().__init__(position, velocity)

        self.r = r
        self.mass = mass
        self.shape = self._make_shape()

    def _make_shape(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.r, (0, 0))
        body = pymunk.Body(self.mass, inertia)
        body.position = self.position

        shape = pymunk.Circle(body, self.r, (0, 0))
        shape.elasticity = 1.0
        shape.friction = 0.0

        return shape

    def draw(self, screen, color=(255, 0, 0), radius=None):
        x, y = self.position
        self.last_pos = (int(x), int(y))

        if not radius:
            radius = self.r

        pygame.draw.circle(screen, color, (int(x), int(y)), radius)

    def draw_trail(self, screen):
        x, y = self.position

        if not self.last_pos:
            pass
        else:
            pygame.draw.line(
                screen, (220, 220, 220), self.last_pos, (int(x), int(y)), 1
            )

        self.last_pos = (int(x), int(y))
