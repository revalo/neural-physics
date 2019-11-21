import pygame
import pymunk

from renderer.entity import Entity


class Rect(Entity):
    # position refers to top left corner
    def __init__(self, position, velocity, height, width, mass=10):
        super(Rect, self).__init__(position, velocity)
        self.mass = mass
        self.height = height
        self.width = width

    def get_vertices(self):
        return (
            (int(self.position[0]), int(self.position[1])),
            (int(self.position[0] + self.width), int(self.position[1])),
            (int(self.position[0] + self.width), int(self.position[1] + self.height)),
            (int(self.position[0]), int(self.position[1] + self.height)),
        )

    def get_shape(self):
        inertia = pymunk.moment_for_poly(self.mass, self.get_vertices())
        body = pymunk.Body(self.mass, inertia)
        body.position = self.position

        shape = pymunk.Poly.create_box(body, size=(self.width, self.height))
        shape.elasticity = 1.0
        shape.friction = 0.0

        return shape

    def draw(self, screen, color=(255, 0, 0)):
        pygame.draw.polygon(screen, color, self.get_vertices())
