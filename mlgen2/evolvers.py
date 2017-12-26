from typing import List, Tuple
import numpy as np
import random

from mlgen2.bases import Physique, Location, Sensory, Brain, Critter


class GlobalGA(object):
    """
    Absorbs all critters of a type and mutates from the best performers.
    """
    def __init__(self,
                 selection_frac: float,
                 crossover_rate: float,
                 physical_mutation_rate: float,
                 mental_mutation_rate: float):
        """

        :param selection_frac:
        :param physical_mutation_rate:
        :param mental_mutation_rate:
        """
        assert 0 < selection_frac < 1
        assert 0 <= crossover_rate <= 1
        assert 0 <= physical_mutation_rate
        assert 0 <= mental_mutation_rate

        self.selection_frac = selection_frac
        self.crossover_rate = crossover_rate
        self.physical_mutation_rate = physical_mutation_rate
        self.mental_mutation_rate = mental_mutation_rate

    def _select(self, critters: List[Critter]) -> List[Critter]:
        """ Selects a subset of critters """
        # sorts critters by fitness
        c = sorted(critters, key=lambda x: x.fitness, reverse=True)
        # return the top number determined by the selection frac
        return c[:int(len(c) * self.selection_frac)]

    def _crossover(self, primary: Critter, secondary: Critter) -> Tuple[Physique, Brain]:
        """
        Creates a new physique and brain from a primary and secondary parent
        using the crossover rate. A crossover rate of 0.5 produces a perfect average
        between the two parents, but for rates close to 0 or 1, the
        """

        def _crossover_att(p: object, s: object, att: str):
            """ applies crossover to a single attribute """
            rate = self.crossover_rate
            return getattr(p, att) * (1 - rate) + (getattr(s, att) * rate)

        # first the physical attributes
        p_atts = Physique().__dict__.keys()
        cross_p_atts = {k: _crossover_att(primary, secondary, k) for k in p_atts}
        physique = Physique(**cross_p_atts)

        # now the mental attributes
        m_atts = ["coefs", "intercepts"]
        cross_b_atts = {k: _crossover_att(primary, secondary, k) for k in m_atts}
        brain = Brain(**cross_b_atts)
        return physique, brain

    def _mutate_physique(self, physique: Physique) -> Physique:
        """ randomly alters one attribute of physique by up to the physical mutation rate"""

        # randomly select one physical attribute to mutate
        att = random.choice(physique.__dict__.keys())
        val = getattr(physique, att)
        new_val = val + np.random.uniform(
            val * (1 - self.physical_mutation_rate),
            val * (1 + self.physical_mutation_rate))
        setattr(physique, att, new_val)
        return physique

    def _mutate_brain(self, brain: Brain) -> Brain:
        """
        randomly alters one layer of either the coeficients OR the intercepts of
        input brain by up to the mental mutation rate
        """
        att = random.choice(["coefs", "intercepts"])
        layer_num = random.choice(list(range(brain.coefs)))
        full_val = getattr(brain, att)
        val = full_val[layer_num]
        new_val = val + np.random.uniform(
            val * (1 - self.mental_mutation_rate),
            val * (1 + self.mental_mutation_rate),
            val.shape)

        full_val[layer_num] = new_val
        setattr(brain, att, full_val)
        return brain

    def mate_pair(self, primary: Critter, secondary: Critter, location: Location,
                  mutate_physique: bool = True, mutate_brain: bool = True,
                  **kwargs) -> Critter:
        """
        Mates a pair of critters to produce a single offspring at input location.

        :param primary: first parent
        :param secondary: second parent
        :param location: location to grant new offspring
        :param mutate_physique: allow physique to mutate
        :param mutate_brain: allow brain to mutate
        :param kwargs: kwargs to pass along to instantiation of new critter
        :return:
        """
        assert primary.__class__ == secondary.__class__

        sensory = primary.sensory
        cls = primary.__class__

        new_physique, new_brain = self._crossover(primary, secondary)
        if mutate_physique:
            new_physique = self._mutate_physique(new_physique)
        if mutate_brain:
            new_brain = self._mutate_brain(new_brain)

        return cls(new_physique, sensory, new_brain, location, **kwargs)

    def evolve_generation(self,
                          critters: List[Critter],
                          mutate_physique: bool = True,
                          mutate_brain: bool = True) -> List[Critter]:
        """
        Returns a new list of critters evolved from the greatest fitness members of
        the input list of critters

        :param critters: List of input critters
        :param mutate_physique: allow physique to mutate
        :param mutate_brain: allow brain to mutate
        :return:
        """
        # get the best performers and set up gen pool
        selection = self._select(critters)
        new_gen_size = len(critters) - len(selection)
        new_gen_pool = list()

        # iteratively mate two random parents from selection pool and add offspring to pool
        for n in range(new_gen_size):
            a, b = random.sample(selection, 2)
            offspring = self.mate_pair(
                a, b, Location(), mutate_physique=mutate_physique, mutate_brain=mutate_brain)
            new_gen_pool.append(offspring)

        return selection + new_gen_pool












