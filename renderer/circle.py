import pygame
import Box2D

from renderer.constants import BOX2D_MUL
from renderer.entity import Entity


class Circle(Entity):
    def __init__(self, position, velocity, r):
        super(Circle, self).__init__(position, velocity)

        self.r = r

    def get_body_def(self):
        bodyDef = Box2D.b2BodyDef(
            position=self.position,
            linearVelocity=self.velocity,
            angle=0.0,
            linearDamping=0.0,
            type=Box2D.b2_dynamicBody,
        )

        return bodyDef

    def get_fixture(self):
        circle = Box2D.b2CircleShape(pos=(0, 0), radius=self.r * BOX2D_MUL)
        fixture = Box2D.b2FixtureDef(shape=circle, density=1.0, restitution=1.0,)

        return fixture

    def draw(self, screen, color=(255, 0, 0)):
        x, y = self.position
        pygame.draw.circle(screen, color, (int(x), int(y)), self.r)
