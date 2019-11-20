import pygame
import pymunk

from renderer.entity import Entity


class Circle(Entity):
    def __init__(self, position, velocity, r, mass=10):
        super(Circle, self).__init__(position, velocity)

        self.r = r
        self.mass = mass

    def get_shape(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.r, (0, 0))
        body = pymunk.Body(self.mass, inertia)
        body.position = self.position

        shape = pymunk.Circle(body, self.r, (0, 0))
        shape.elasticity = 1.0
        shape.friction = 0.0

        return shape

    def draw(self, screen, color=(255, 0, 0)):
        x, y = self.position
        pygame.draw.circle(screen, color, (int(x), int(y)), self.r)
