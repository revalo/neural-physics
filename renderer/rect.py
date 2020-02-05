import pygame
import pymunk

from renderer.entity import Entity


class Rect(Entity):
    # position refers to the center of the box
    def __init__(self, position, velocity, height, width, mass=10):
        super().__init__(position, velocity)
        self.mass = mass
        self.height = height
        self.width = width
        self.shape = self._make_shape()

    def get_vertices(self):
        """
        Returns the vertices of the box in world coordinates
        in a counterclockwise winding
        """
        vertices = []
        for v in self.shape.get_vertices():
            x, y = v.rotated(self.shape.body.angle) + self.position
            vertices.append((x, y))
        return vertices

    def _make_shape(self):
        inertia = pymunk.moment_for_box(self.mass, (self.width, self.height))
        body = pymunk.Body(self.mass, inertia)
        body.position = self.position

        shape = pymunk.Poly.create_box(body, size=(self.width, self.height))
        shape.elasticity = 0.5
        shape.friction = 0.4
        # shape.elasticity = 1.0
        # shape.friction = 0.0

        return shape

    def draw(self, screen, color=(255, 0, 0)):
        pygame.draw.polygon(screen, color, self.get_vertices())
