import pygame
import pymunk

from renderer.entity import Entity


class Wall(Entity):
    def __init__(self, position, w, h, elasticity=0.8, friction=0.4):
        super(Wall, self).__init__(position, (0, 0))
        self.w = w
        self.h = h
        self.elasticity = elasticity
        self.friction = friction
        self.shape = self._make_shape()

    def _make_shape(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = self.position

        box = pymunk.Poly.create_box(body, size=(self.w, self.h))
        box.friction = self.friction
        box.elasticity = self.elasticity

        return box

    def draw(self, screen, color=(255, 0, 0)):
        x, y = self.position
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(int(x - self.w / 2), int(y - self.h / 2), self.w, self.h),
        )
