"""Generator classes providing points using sampling"""

import numpy as np

from libensemble.generators import Generator, LibensembleGenerator
from libensemble.utils.misc import list_dicts_to_np

__all__ = [
    "UniformSample",
    "UniformSampleDicts",
]


class SampleBase(LibensembleGenerator):
    """Base class for sampling generators"""

    def _get_user_params(self, user_specs):
        """Extract user params"""
        self.ub = user_specs["ub"]
        self.lb = user_specs["lb"]
        self.n = len(self.lb)  # dimension
        assert isinstance(self.n, int), "Dimension must be an integer"
        assert isinstance(self.lb, np.ndarray), "lb must be a numpy array"
        assert isinstance(self.ub, np.ndarray), "ub must be a numpy array"


class UniformSample(SampleBase):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.
    """

    def __init__(self, variables: dict, objectives: dict, _=[], persis_info={}, gen_specs={}, libE_info=None, **kwargs):
        super().__init__(variables, objectives, _, persis_info, gen_specs, libE_info, **kwargs)
        self._get_user_params(self.gen_specs["user"])

    def ask_numpy(self, n_trials):
        return list_dicts_to_np(
            UniformSampleDicts(
                self.variables, self.objectives, self.History, self.persis_info, self.gen_specs, self.libE_info
            ).ask(n_trials)
        )

    def tell_numpy(self, calc_in):
        pass  # random sample so nothing to tell


# List of dictionaries format for ask (constructor currently using numpy still)
# Mostly standard generator interface for libE generators will use the ask/tell wrappers
# to the classes above. This is for testing a function written directly with that interface.
class UniformSampleDicts(Generator):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.

    This currently adheres to the complete standard.
    """

    def __init__(self, variables: dict, objectives: dict, _, persis_info, gen_specs, libE_info=None, **kwargs):
        self.variables = variables
        self.gen_specs = gen_specs
        self.persis_info = persis_info

    def ask(self, n_trials):
        H_o = []
        for _ in range(n_trials):
            trial = {}
            for key in self.variables.keys():
                trial[key] = self.persis_info["rand_stream"].uniform(self.variables[key][0], self.variables[key][1])
            H_o.append(trial)
        return H_o

    def tell(self, calc_in):
        pass  # random sample so nothing to tell
