from typing import Tuple


class World(object):
    """
    the world that the critters inhabit
    """
    def __init__(self, size: Tuple):

        self.size = size

        self.objects = list()
        self.age = 0
        self.step = 1