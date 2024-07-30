from typing import List
from dd.autoref import BDD

MAXIMUM_CONFIDENCE = 1.0  # maximum confidence for rejection
ACCEPTANCE_CONFIDENCE = 0.0  # confidence when accepting
INCREDIBLE_CONFIDENCE = 1.0  # confidence when rejecting due to incredibility
SKIPPED_CONFIDENCE_NOVELTY_MODE = -1.0  # confidence when training novelties (has no meaning)
SKIPPED_CONFIDENCE = 1.0  # confidence when no distance is used
CONVEX_HULL_HALF_SPACE_DISTANCE_CORNER_CASE = 0.0  # half-space confidence for flat convex hulls
COMPOSITE_ABSTRACTION_POLICY = 2  # policy for CompositeAbstraction and multi-layer monitors; possible values:


def _var(i):
    return 'x' + str(i)


class BinaryMonitor(object):
    def __init__(self, gamma=0):
        self.bdd = BDD()
        self.formula = None
        self.dim = -1
        self.gamma = gamma

    def initialize(self, n_watched_neurons):
        self.dim = n_watched_neurons
        for i in range(n_watched_neurons):
            self.bdd.declare(_var(i))

    def create(self, vector):
        self.add(vector, create=True)

    def add(self, vector, create=False):
        a = self._to_bit_vector(vector)
        conjunction = self._to_conjunction(a)
        f = self.bdd.add_expr(conjunction)
        g = self.formula
        if g is None:
            g = f
        else:
            g = g | f
        for _ in range(self.gamma):
            g_prev = g
            for i in range(self.dim):
                g = g | self.bdd.exist([_var(i)], g_prev)
        self.formula = g

    def query(self, vector):
        #a = self._to_bit_vector(vector)
        a = vector
        conjunction = self._to_conjunction(a)
        f = self.bdd.add_expr(conjunction)
        g = self.formula & f
        accepts = self.bdd.pick(g) is not None
        if accepts:
          return 0
        else:
          return 1

    def make_verdicts(self, features):
        query_results = [self.query(x) for x in features]
        return query_results

    @staticmethod
    def _to_bit_vector(vector):
        a = [0] * len(vector)
        for i, v in enumerate(vector):
            if v <= 0:
                a[i] = 0
            else:
                a[i] = 1
        return a

    @staticmethod
    def _to_conjunction(a):
        conjunction = ""
        for i, v in enumerate(a):
            if i > 0:
                conjunction += " & "
            if v == 0:
                conjunction += "~x"
            else:
                conjunction += "x"
            conjunction += str(i)
        return conjunction

    def clear(self):
        self.bdd = BDD()
        self.formula = None
        assert self.dim > 0, "Boolean abstraction has not been initialized yet."
        self.initialize(self.dim)

    def add_finalized(self, vector):
        self.add(vector)

    @staticmethod
    def default_options():
        gamma = 0
        return gamma,  # needs to be a tuple
