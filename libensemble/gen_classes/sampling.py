"""Generator classes providing points using sampling"""

import numpy as np

from libensemble import Generator

__all__ = [
    "UniformSample",
]


class UniformSample(Generator):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.
    """

    def __init__(self, _, persis_info, gen_specs, libE_info=None) -> list:
        # self.H = H
        self.persis_info = persis_info
        self.gen_specs = gen_specs
        self.libE_info = libE_info
        self._get_user_params(self.gen_specs["user"])

    def ask(self, n_trials):
        H_o = self._ask_np(n_trials)
        return [{"x": x.tolist()} for x in H_o["x"]]

    def _ask_np(self, n_trials):
        H_o = np.zeros(n_trials, dtype=self.gen_specs["out"])
        H_o["x"] = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (n_trials, self.n))

        if "obj_component" in H_o.dtype.fields:  # needs H_o - needs to be created in here.
            H_o["obj_component"] = self.persis_info["rand_stream"].integers(
                low=0, high=self.gen_specs["user"]["num_components"], size=n_trials
            )
        return H_o

    def tell(self, calc_in):
        pass  # random sample so nothing to tell

    def _tell_np(self, calc_in):
        self.tell(calc_in)

    def _get_user_params(self, user_specs):
        """Extract user params"""
        # b = user_specs["initial_batch_size"]
        self.ub = user_specs["ub"]
        self.lb = user_specs["lb"]
        self.n = len(self.lb)  # dimension
        assert isinstance(self.n, int), "Dimension must be an integer"
        assert isinstance(self.lb, np.ndarray), "lb must be a numpy array"
        assert isinstance(self.ub, np.ndarray), "ub must be a numpy array"
