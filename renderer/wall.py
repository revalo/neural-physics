import pygame
import Box2D

from constants import BOX2D_MUL
from entity import Entity


class Wall(Entity):
    def __init__(self, position, w, h):
        super(Wall, self).__init__(position, (0, 0))
        self.w = w
        self.h = h

    def get_body_def(self):
        bodyDef = Box2D.b2BodyDef(
            position=self.position,
            linearVelocity=self.velocity,
            angle=0.0,
            linearDamping=0.0,
            type=Box2D.b2_staticBody,
        )

        return bodyDef

    def get_fixture(self):
        box = Box2D.b2PolygonShape(box=(self.w / 2 * BOX2D_MUL, self.h / 2 * BOX2D_MUL))
        fixture = Box2D.b2FixtureDef(shape=box, density=1.0, restitution=1.0,)

        return fixture

    def draw(self, screen, color=(255, 0, 0)):
        x, y = self.position
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(int(x - self.w / 2), int(y - self.h / 2), self.w, self.h),
        )
