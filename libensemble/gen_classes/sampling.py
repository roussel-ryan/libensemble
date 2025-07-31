"""Generator classes providing points using sampling"""

import numpy as np
from generator_standard import Generator
from generator_standard.vocs import VOCS

from libensemble.generators import LibensembleGenerator

__all__ = [
    "UniformSample",
    "StandardSample",
]


class UniformSample(LibensembleGenerator):
    """
    Samples over the domain specified in the VOCS.
    """

    def __init__(self, VOCS: VOCS, variables_mapping: dict = {}):
        super().__init__(VOCS, variables_mapping=variables_mapping)
        self.rng = np.random.default_rng(1)

        self.np_dtype = []
        for i in self.variables_mapping.keys():
            self.np_dtype.append((i, float, (len(self.variables_mapping[i]),)))

        self.n = len(list(self.VOCS.variables.keys()))
        self.lb = np.array([VOCS.variables[i].domain[0] for i in VOCS.variables])
        self.ub = np.array([VOCS.variables[i].domain[1] for i in VOCS.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        for i in range(n_trials):
            for key in self.variables_mapping.keys():
                out[i][key] = self.rng.uniform(self.lb, self.ub, (self.n))

        return out

    def ingest_numpy(self, calc_in):
        pass  # random sample so nothing to tell


class StandardSample(Generator):
    """
    This sampler only adheres to the complete standard interface, with no additional numpy methods.
    """

    def __init__(self, VOCS: VOCS):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        super().__init__(VOCS)

    def _validate_vocs(self, VOCS):
        assert len(self.VOCS.variables), "VOCS must contain variables."

    def suggest(self, n_trials):
        output = []
        for _ in range(n_trials):
            trial = {}
            for key in self.VOCS.variables.keys():
                trial[key] = self.rng.uniform(self.VOCS.variables[key].domain[0], self.VOCS.variables[key].domain[1])
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell
