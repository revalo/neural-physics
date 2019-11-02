"""Renderer does physics simulations and rendering.
"""

from constants import BOX2D_MUL, PHYSICS_MUL

from wall import Wall

import Box2D


class World(object):
    def __init__(self, width=500, height=500, gravity=(0, 0)):
        self.width = width
        self.height = height

        self.world = Box2D.b2World(gravity=gravity, doSleep=True)

        self.entities = {}
        self.bodies = {}

        self.id_counter = 0

        # Add invisible walls at edges.
        self.add_entity(Wall((-5, self.height / 2), 10, self.height))
        self.add_entity(Wall((self.width + 10, self.height / 2), 10, self.height))
        self.add_entity(Wall((self.width / 2, -5), self.width, 10))
        self.add_entity(Wall((self.width / 2, self.height + 10), self.width, 10))

    def add_entity(self, entity):
        body_def = entity.get_body_def()
        body = self.world.CreateBody(body_def)
        body.CreateFixture(entity.get_fixture())

        entity.id = self.id_counter

        self.bodies[entity.id] = body
        self.entities[entity.id] = entity

        self.id_counter += 1

    def remove_entity(self, entity):
        body = self.bodies[entity.id]

        self.world.destroyBody(body)

        del self.bodies[entity.id]
        del self.entities[entity.id]

    def copy_to_entity(self, entity):
        body = self.bodies[entity.id]

        x, y = body.position

        entity.position = (x * PHYSICS_MUL, y * PHYSICS_MUL)
        entity.velocity = body.linearVelocity

    def copy_from_entity(self, entity):
        body = self.bodies[entity.id]

        x, y = entity.position

        body.position = (x * BOX2D_MUL, y * BOX2D_MUL)
        body.linearVelocity = entity.velocity

    def step(self, dt=0.01):
        for entity in self.entities.values():
            self.copy_from_entity(entity)

        self.world.Step(dt, 8, 3)

        for entity in self.entities.values():
            self.copy_to_entity(entity)
