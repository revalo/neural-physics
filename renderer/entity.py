import abc


class Entity(abc.ABC):
    @abc.abstractmethod
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.id = -1

    @abc.abstractmethod
    def _make_shape(self):
        """
        Makes the pymunk.Shape object that the entity is associated with

        Should be called in __init__, and generally called at the end
        after other fields are set
        """
        pass

    def get_shape(self):
        return self.shape

    @abc.abstractmethod
    def draw(self, screen, color=(255, 0, 0)):
        pass
