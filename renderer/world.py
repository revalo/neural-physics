"""Renderer does physics simulations and rendering.
"""

from renderer.wall import Wall

import pymunk


class World(object):
    def __init__(self, width=500, height=500, gravity=(0, 0), wall_elasticity=0.8):
        self.width = width
        self.height = height

        self.space = pymunk.Space()
        self.space.gravity = gravity

        self.entities = {}
        self.shapes = {}

        self.id_counter = 0

        # Add invisible walls at edges.
        self.add_entity(Wall((-5, self.height / 2), 10, self.height, wall_elasticity))
        self.add_entity(
            Wall((self.width + 5, self.height / 2), 10, self.height, wall_elasticity)
        )
        self.add_entity(Wall((self.width / 2, -5), self.width, 10, wall_elasticity))
        self.add_entity(
            Wall((self.width / 2, self.height + 5), self.width, 10, wall_elasticity)
        )

    def add_entity(self, entity):
        shape = entity.get_shape()
        self.space.add(shape.body, shape)

        entity.id = self.id_counter

        self.shapes[entity.id] = shape
        self.entities[entity.id] = entity

        self.id_counter += 1

    def remove_entity(self, entity):
        shape = self.shapes[entity.id]

        self.space.remove(shape.body, shape)

        del self.shapes[entity.id]
        del self.entities[entity.id]

    def copy_to_entity(self, entity):
        shape = self.shapes[entity.id]

        entity.position = shape.body.position
        entity.velocity = shape.body.velocity

    def copy_from_entity(self, entity):
        shape = self.shapes[entity.id]

        shape.body.position = entity.position
        shape.body.velocity = entity.velocity

    def step(self, dt=0.01):
        for entity in self.entities.values():
            self.copy_from_entity(entity)

        self.space.step(dt)

        for entity in self.entities.values():
            self.copy_to_entity(entity)
