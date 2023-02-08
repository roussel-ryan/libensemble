import inspect
import logging
import logging.handlers

import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG

logger = logging.getLogger(__name__)


class Runners:
    """Determines and returns methods for workers to run user functions.

    Currently supported: direct-call and funcX
    """

    def __init__(self, sim_specs, gen_specs):
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.sim_f = sim_specs["sim_f"]
        self.gen_f = gen_specs.get("gen_f")
        self.has_funcx_sim = len(sim_specs.get("funcx_endpoint", "")) > 0
        self.has_funcx_gen = len(gen_specs.get("funcx_endpoint", "")) > 0
        self.funcx_exctr = None

        if any([self.has_funcx_sim, self.has_funcx_gen]):
            try:
                from funcx import FuncXClient
                from funcx.sdk.executor import FuncXExecutor

                self.funcx_exctr = FuncXExecutor(FuncXClient())

            except ModuleNotFoundError:
                logger.warning("funcX use detected but funcX not importable. Is it installed?")

    def make_runners(self):
        """Creates functions to run a sim or gen. These functions are either
        called directly by the worker or submitted to a funcX endpoint."""

        def run_sim(calc_in, persis_info, libE_info):
            """Determines how to run sim."""
            if self.has_funcx_sim and self.funcx_exctr:
                result = self._funcx_result
            else:
                result = self._normal_result

            return result(calc_in, persis_info, self.sim_specs, libE_info, self.sim_f)

        if self.gen_specs:

            def run_gen(calc_in, persis_info, libE_info):
                """Determines how to run gen."""
                if self.has_funcx_gen and self.funcx_exctr:
                    result = self._funcx_result
                else:
                    result = self._normal_result

                return result(calc_in, persis_info, self.gen_specs, libE_info, self.gen_f)

        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    @staticmethod
    def _is_fx_wrapped(user_f):
        return hasattr(user_f, "__wrapped__") and hasattr(user_f.__wrapped__, "unpacking")

    @staticmethod
    def _unpacked_repacked_result(nparams, args, calc_in, specs, user_f):
        outfield = specs["out"][0][0]  # hopefully only outfield
        calc_in = calc_in.item()[0]  # unpack
        H_o = np.zeros(1, dtype=specs["out"])  # repacking

        out = user_f(*args[:nparams])  # call
        if not isinstance(out, tuple):  # got one return
            H_o[outfield] = out
            return H_o
        else:  # multiple returns, adjust first to be repacked array
            H_o[outfield] = out[0]
            return (H_o,) + out[1:]

    def _normal_result(self, calc_in, persis_info, specs, libE_info, user_f):
        """User function called in-place"""
        nparams = len(inspect.signature(user_f).parameters)
        args = [calc_in, persis_info, specs, libE_info]

        if self._is_fx_wrapped(user_f):  # wrapped function is tagged
            return self._unpacked_repacked_result(nparams, args, calc_in, specs, user_f)
        else:
            return user_f(*args[:nparams])

    def _funcx_result(self, calc_in, persis_info, specs, libE_info, user_f):
        """User function submitted to funcX"""
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        future = self.funcx_exctr.submit(
            user_f,
            calc_in,
            persis_info,
            specs,
            libE_info,
            endpoint_id=specs["funcx_endpoint"],
        )
        remote_exc = future.exception()  # blocks until exception or None
        if remote_exc is None:
            return future.result()
        else:
            raise remote_exc
