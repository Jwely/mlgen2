import numpy as np
from matplotlib.axes import Axes

from mlgen2.environments import World
from mlgen2.bases import Critter, Physique, Location


class Plant(Critter):
    """ plants are spawned with 1 energy and just sit there."""
    def __init__(self, location: Location):
        super().__init__(None, None, None, location, energy=1)

        # heading is meaningless on plants, but this makes them plot uniformly
        self.location.heading = 1.5 * np.pi()

    def plot(self, ax: Axes):
        """ plots a plant on input axis """
        position_kwargs = dict(radius=.05, vertices=3, orientation=0,
                               facecolor="lightgreen", edgecolor="darkgreen", zorder=5)
        heading_kwargs = dict(hlen=0.1, color="brown", linewidth=1, zorder=4)
        super()._plot(ax, position_kwargs, heading_kwargs)

    def simulate(self, world_dat: World):
        """ do nothing! I'm a plant"""
        pass


class Herby(Critter):
    """ A Herby is an herbivore, they eat plants mostly. """

    def plot(self, ax: Axes, position_kwargs, heading_kwargs):
        """ plots a herby on input axis """
        default_position_kwargs = dict(
            radius=0.05, vertices=10, facecolor="lightblue", edgecolor="black")
        default_position_kwargs.update(position_kwargs)

        default_heading_kwargs = dict(hlen=0.08, linewidth=1, color="black", zorder=11)
        default_heading_kwargs.update(heading_kwargs)
        super()._plot(ax, default_position_kwargs, default_heading_kwargs)