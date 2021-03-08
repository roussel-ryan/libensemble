# """
# Runs libEnsemble with Surmise calibration test.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_persistent_surmise_killsims.py
#    python3 test_persistent_surmise_killsims.py --nworkers 3 --comms local
#    python3 test_persistent_surmise_killsims.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

# Requires:
#   Install Surmise package


# NOTE (REMOVE WHEN FIXED. **********************************************
# TODO for step 1:
#    Rename files/vars as required.
#    Determine pass condition for test (assertions at end).

import os

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.gen_funcs.persistent_surmise_calib import testcalib as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.sim_funcs.borehole_kills import borehole as sim_f
from libensemble.tests.regression_tests.common import build_borehole  # current location
from libensemble.executors.executor import Executor
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# from libensemble import libE_logger
# libE_logger.set_level('DEBUG')  # To get debug logging in ensemble.log

if __name__ == '__main__':

    nworkers, is_manager, libE_specs, _ = parse_args()

    n_init_thetas = 15              # Initial batch of thetas
    n_x = 5                         # No. of x values
    nparams = 4                     # No. of theta params
    ndims = 3                       # No. of x co-ordinates.
    max_add_thetas = 20             # Max no. of thetas added for evaluation
    step_add_theta = 10             # No. of thetas to generate per step, before emulator is rebuilt
    n_explore_theta = 200           # No. of thetas to explore while selecting the next theta
    obsvar = 10 ** (-1)             # Constant for generating noise in obs

    # Batch mode until after batch_sim_id (add one theta to batch for observations)
    batch_sim_id = (n_init_thetas + 1) * n_x

    # Stop after max_emul_runs runs of the emulator
    max_evals = batch_sim_id + max_add_thetas*n_x

    sim_app = os.path.join(os.getcwd(), "borehole.x")
    if not os.path.isfile(sim_app):
        build_borehole()

    exctr = Executor()  # Run serial sub-process in place
    exctr.register_calc(full_path=sim_app, app_name='borehole')

    # Subprocess variant creates input and output files for each sim
    libE_specs['sim_dirs_make'] = True  # To keep all - make sim dirs
    # libE_specs['use_worker_dirs'] = True  # To overwrite - make worker dirs only

    # Rename ensemble dir for non-inteference with other regression tests
    libE_specs['ensemble_dir_path'] = 'ensemble_calib_kills'

    sim_specs = {'sim_f': sim_f,
                 'in': ['x', 'thetas'],
                 'out': [('f', float)],
                 'user': {'num_obs': n_x,
                          'batch_to_sim_id': batch_sim_id} #SH change name batch_to_sim_id...
                 }

    gen_out = [('x', float, ndims), ('thetas', float, nparams),
               ('priority', int), ('obs', float, n_x), ('obsvar', float, n_x)]

    gen_specs = {'gen_f': gen_f,
                 'in': [o[0] for o in gen_out]+['f', 'returned'],
                 'out': gen_out,
                 'user': {'n_init_thetas': n_init_thetas,        # Num thetas in initial batch
                          'num_x_vals': n_x,                     # Num x points to create
                          'step_add_theta': step_add_theta,      # No. of thetas to generate per step
                          'n_explore_theta': n_explore_theta,    # No. of thetas to explore each step
                          'obsvar': obsvar,                      # Variance for generating noise in obs
                          'batch_to_sim_id': batch_sim_id,       # Up to this sim_id, wait for all results to return.
                          'ignore_cancelled': True,              # Do not use returned results that have been cancelled
                          'priorloc': 1,                         # Prior location in the unit cube.
                          'priorscale': 0.2,                     # Standard deviation of prior
                          }
                 }

    # alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {'batch_mode': True}}
    alloc_specs = {'alloc_f': alloc_f,
                   'out': [('given_back', bool)],
                   'user': {'batch_to_sim_id': batch_sim_id,
                            'batch_mode': False  # set batch mode (alloc behavior after batch_sim_id)
                            }
                   }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Currently just allow gen to exit if mse goes below threshold value
    exit_criteria = {'sim_max': max_evals}
    # exit_criteria = {'sim_max': max_evals, 'stop_val': ('mse', mse_exit)}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs,
                                exit_criteria, persis_info,
                                alloc_specs=alloc_specs,
                                libE_specs=libE_specs)

    if is_manager:
        print('Cancelled sims', H['sim_id'][H['cancel']])
        print('Killed sims', H['sim_id'][H['kill_sent']])
        # MC: Clean up of unreturned
        # assert np.all(H['returned'])  #SH Could be all either returned or cancelled.
        save_libE_output(H, persis_info, __file__, nworkers)
