"""
This file contains the definition of the success probability and the
Werner parameter of protocol-units, including
entanglement swap, entanglement distillation and cut-off
(Table 1 in the article)
"""

import numba as nb
import numpy as np


__all__ = [
    "memory_cut_off", "fidelity_cut_off",
    "get_one", "get_swap_wout",
    "get_dist_prob_suc", "get_dist_prob_fail", "get_dist_prob_wout"
]

########################################################################
"""
Success probability p and
the resulting Werner parameter of swap and distillation.

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
w1, w2: float
    The Werner parameter of the two input links.
decay_factor: float
    exp(- |t1 - t2| / t_coh)

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def get_one(t1, t2, w1, w2, decay_factor):
    """
    Get a trivial one
    """
    return 1.


@nb.jit(nopython=True, error_model="numpy")
def get_swap_wout(t1, t2, w1, w2, decay_factor):
    """
    Get w_swap
    """
    return w1 * w2 * decay_factor


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_wout(t1, t2, w1, w2, decay_factor):
    """
    Get p_dist * w_dist
    """
    if t1 < t2:
        w1 *= decay_factor
    else:
        w2 *= decay_factor
    return 1./6. * (w1 + w2 + 4 * w1 * w2)


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_fail(t1, t2, w1, w2, decay_factor):
    """
    Get 1 - p_dist
    """
    return 1. - get_dist_prob_suc(t1, t2, w1, w2, decay_factor)


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_suc(t1, t2, w1, w2, decay_factor):
    """
    Get p_dist
    """
    return 0.5 + 0.5 * w1 * w2 * decay_factor


########################################################################
"""
Cut-off functions

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
w1, w2: float
    The Werner parameter of the two input links.
mt_cut: int
    The memory time cut-off.
w_cut: float
    The Werner parameter cut-off. (0 < w_cut < 1)
    Set a cut-off on the input links's Werner parameter
t_coh: int or float
    The memory coherence time.

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def memory_cut_off(
        t1, t2, w1=1.0, w2=1.0,
        mt_cut=np.iinfo(np.int).max, w_cut=1.0, t_coh=0, ycut=True):
    """
    Memory storage cut-off. The two input links suvives only if
    |t1-t2|<=mt_cut
    """
    if abs(t1 - t2) > mt_cut:
        # constant shift mt_cut is added in the iterative convolution
        return min(t1, t2), False
    else:
        return max(t1, t2), True

