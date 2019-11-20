import abc


class Entity(abc.ABC):
    @abc.abstractmethod
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.id = -1

    @abc.abstractmethod
    def get_shape(self):
        pass

    @abc.abstractmethod
    def draw(self, screen, color=(255, 0, 0)):
        pass
