from typing import Tuple, List, Union
import numpy as np

from matplotlib.patches import CirclePolygon
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from math import sin
from math import cos

from mlgen2.environments import World


class Physique(object):
    """
    The physical attributes of an entity that define it's interface
    with the world
    """
    def __init__(self,
                 mass: float = None,
                 max_s: float = None,
                 max_dsp: float = None,
                 max_dsn: float = None,
                 max_dh: float = None):

        # attributes unchanged for critters life
        self.mass = mass        # governs energy expenditure and consumption requirements
        self.max_s = max_s      # maximum absolute speed
        self.max_dsp = max_dsp  # maximum positive change in speed (ds)  accel
        self.max_dsn = max_dsn  # maximum negative change in speed (ds)  deccel
        self.max_dh = max_dh    # maximum change in heading (rotation)


class Location(object):

    def __init__(self,
                 position: Tuple[int, int] = (0,0),
                 heading: float = 0,
                 speed: float = 0,
                 ):
        # attributes that change over the critters life
        self.position = position  # position vector (x,y)
        self.heading = heading  # direction of facing
        self.speed = speed  # speed of movement to heading


class Sensory(object):
    """
    When called, extracts all relevant data about the world, and about the
    critters self, then outputs arrays of data for feeding into neural network.
    """
    def __init__(self, labels: List[str] = list()):
        self.labels = labels
        self.width = len(labels)

    def __call__(self, world_dat: World, physique: Physique) -> np.array:
        raise Exception("Base Sensory class __call__ should be overriden")


class Brain(object):
    """ ultra simplified neural net, no back-propagation """
    def __init__(self,
                 afs: List[np.array],
                 coefs: List[np.array],
                 intercepts: List[np.array]):
        """
        All of the
        :param afs: list of activation functions between each layer
        :param coefs: list of coefficient matrices between each layer
        :param intercepts: list of intercept matrices between each layer
        """
        assert len(afs) == len(coefs) == len(intercepts)

        self.afs = afs                      # activation functions
        self.coefs = coefs                  # coefficients (mult)
        self.intercepts = intercepts        # intercepts (add)

        self._n_layers = len(coefs) + 1     # includes input and output layer

    @classmethod
    def random_init(cls, af, lws: Tuple):
        """
        With layer dimensions, randomizes coefficients and intercepts in NN
        :param af:
        :param lws:
        :return:
        """

        coefs = list()
        ints = list()

        for i in range(len(lws) - 1):
            coefs.append(np.random.rand(lws[i], lws[i + 1]))
            ints.append(np.random.rand(lws[i + 1]))

        return cls(af, coefs, ints)

    def __call__(self, x: np.array) -> np.array:
        """ forward propagates input X through network to create output """

        # if a vectorized inputs passed, return vectorized outputs
        if len(x.shape) > 1:
            return np.array([self.__call__(x_) for x_ in x]).flatten()

        # set up activations
        activity = [0 for x in range(self._n_layers)]
        activity[0] = x

        # forward propagate through layers
        for i, (m, b) in enumerate(zip(self.coefs, self.intercepts)):
            a = self.afs[i](np.dot(activity[i], m) + b)
            activity[i + 1] = a

        # output activity on the last node (output neuron)
        return activity[-1]


class Critter(object):
    def __init__(self,
                 physique: Union[Physique, None],
                 sensory: Union[Sensory, None],
                 brain: Union[Brain, None],
                 location: Location,
                 fitness: int = 0,
                 energy: float = 0):

        # lifetime attributes of critter
        self.physique = physique
        self.sensory = sensory
        self.brain = brain

        # dynamic attributes of critter
        self.location = location
        self.fitness = fitness
        self.energy = energy

    def simulate(self, world_dat: World):
        """ simulates a critter for one time step """
        raise Exception("Base Critter simulate method should be overriden ")

    def _plot(self, ax: Axes, position_kwargs: dict, heading_kwargs: dict):
        """ adds a representation of this critter to matplotlib axis object

        position_kwargs should likely have keys [radius, zorder, facecolor, edgecolor]
        heading_kwargs should likely have keys [color, linewidth, zorder]
        """

        # position
        p0 = self.location.position
        patch = CirclePolygon(p0, **position_kwargs)
        ax.add_artist(patch)

        # heading indicator
        if "hlen" not in heading_kwargs.keys():
            hlen = 0.08
        else:
            hlen = heading_kwargs["hlen"]

        p1 = (cos(self.location.heading) * hlen + p0[0],
              sin(self.location.heading) * hlen + p0[1])
        line = Line2D(p0, p1, **heading_kwargs)
        ax.add_line(line)
        return ax
