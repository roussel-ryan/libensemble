import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def finite_diff_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start 1 persistent generator.  If all points requested by
    the persistent generator for a given (x_ind,f_ind) pair have been returned from the
    simulation evaluation, then this information is given back to the
    persistent generator (where x_ind is in range(n) and f_ind is in range(p))

    .. seealso::
        `test_persistent_fd_param_finder.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_fd_param_finder.py>`_ # noqa
    """

    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})

    support = AllocSupport(W, H, persis_info, sched_opts)
    Work = {}
    gen_count = support.count_persis_gens()

    gen_return_fields = list(set(gen_specs['in'] + sim_specs['in'] +
                                 [n[0] for n in sim_specs['out']] + [('sim_id')]))

    if len(H) and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    # If wid is in persistent mode, and all of its calculated values have
    # returned, give them back to wid. Otherwise, give nothing to wid
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG):

        # What (x_ind, f_ind) pairs have all of the evaluation of all n_ind
        # values complete.
        inds_not_sent_back = ~H['given_back']
        H_tmp = H[inds_not_sent_back]

        inds_to_send = np.array([], dtype=int)
        for x_ind in range(gen_specs['user']['n']):
            for f_ind in range(gen_specs['user']['p']):
                inds = np.logical_and.reduce((H_tmp['x_ind'] == x_ind, H_tmp['f_ind'] == f_ind, H_tmp['returned']))
                if sum(inds) == gen_specs['user']['nf']:
                    inds_to_send = np.append(inds_to_send, H_tmp['sim_id'][inds])

        if len(inds_to_send):
            support.gen_work(Work, wid, gen_return_fields, inds_to_send, persis_info.get(wid), persistent=True)

    points_to_evaluate = ~H['given'] & ~H['cancel_requested']
    for wid in support.avail_worker_ids(persistent=False):
        if np.any(points_to_evaluate):
            # perform sim evaluations (if they exist in History).
            sim_ids_to_send = np.nonzero(points_to_evaluate)[0][0]  # oldest point
            try:
                support.sim_work(Work, wid, H, sim_specs['in'], sim_ids_to_send, persis_info.get(wid))
            except InsufficientFreeResources:
                break
            points_to_evaluate[sim_ids_to_send] = False

        elif gen_count == 0:
            # Finally, call a persistent generator as there is nothing else to do.
            # SH TODO - MAYBE UNNEC - COULD ASSUME ZERO RESOURCES WITH rset_team=[]
            try:
                support.gen_work(Work, wid, gen_specs['in'], [], persis_info.get(wid),
                                 persistent=True)
            except InsufficientFreeResources:
                break
            gen_count += 1

    del support
    return Work, persis_info, 0
