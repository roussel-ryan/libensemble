import numpy as np
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given to be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``alloc_specs['user']['num_active_gen']`` active generators. Also allows
    for a ``'batch_mode'``.

    .. seealso::
        `test_old_aposmm_with_gradients.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_old_aposmm_with_gradients.py>`_ # noqa
    """

    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})

    support = AllocSupport(W, H, persis_info, sched_opts)
    Work = {}
    gen_count = support.count_gens()

    for wid in support.avail_worker_ids():
        # Skip any cancelled points
        # SH TODO - IF USING PAUSED - this might want to check not paused also
        while persis_info['next_to_give'] < len(H) and H[persis_info['next_to_give']]['cancel_requested']:
            persis_info['next_to_give'] += 1

        # Find indices of H that are not yet allocated
        if persis_info['next_to_give'] < len(H):
            # Give sim work if possible
            try:
                support.sim_work(Work, wid, H, sim_specs['in'], [persis_info['next_to_give']], [])
            except InsufficientFreeResources:
                break
            persis_info['next_to_give'] += 1

        elif gen_count < user.get('num_active_gens', gen_count+1):
            lw = persis_info['last_worker']

            last_size = persis_info.get('last_size')

            if len(H):
                # Don't give gen instances in batch mode if points are unfinished
                if user.get('batch_mode') and not support.all_returned(pt_filter=~H['paused'], low_bound=last_size):
                    break
                # Don't call APOSMM if there are runs going but none need advancing
                if len(persis_info[lw]['run_order']):
                    runs_needing_to_advance = np.zeros(len(persis_info[lw]['run_order']), dtype=bool)
                    for run, inds in enumerate(persis_info[lw]['run_order'].values()):
                        runs_needing_to_advance[run] = H['returned'][inds[-1]]

                    if not np.any(runs_needing_to_advance):
                        break

            # Give gen work
            try:
                support.gen_work(Work, wid, gen_specs['in'], range(len(H)), persis_info[lw])
            except InsufficientFreeResources:
                break

            gen_count += 1
            persis_info['total_gen_calls'] += 1
            persis_info['last_worker'] = wid
            persis_info['last_size'] = len(H)

    del support
    return Work, persis_info
