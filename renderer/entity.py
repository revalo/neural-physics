import abc


class Entity(abc.ABC):
    @abc.abstractmethod
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.id = -1

    def get_shape(self):
        return self.shape

    @abc.abstractmethod
    def draw(self, screen, color=(255, 0, 0)):
        pass
