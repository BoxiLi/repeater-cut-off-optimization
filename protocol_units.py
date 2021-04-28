import numba as nb
import numpy as np


__all__ = [
    "memory_cut_off", "fidelity_cut_off",
    "get_one", "get_swap_wout",
    "get_dist_prob_suc", "get_dist_prob_fail", "get_dist_prob_wout"
]


"""
This module contain the defined success probability and
resulting werner parameter of each protocol unit.
(see Table 1 in the article)
"""
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
        mt_cut=np.iinfo(np.int).max, w_cut=1.e-8, rt_cut=np.iinfo(np.int).max, t_coh=np.iinfo(np.int).max):
    """
    Memory storage cut-off. The two input links suvives only if
    |t1-t2|<=mt_cut
    """
    if abs(t1 - t2) > mt_cut:
        # constant shift mt_cut is added in the iterative convolution
        return min(t1, t2), False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def fidelity_cut_off(
    t1, t2, w1, w2,
    mt_cut=np.iinfo(np.int).max, w_cut=1.e-8, rt_cut=np.iinfo(np.int).max, t_coh=np.iinfo(np.int).max):
    """
    Fidelity-dependent cut-off, The two input links suvives only if
    w1 <= w_cut and w2 <= w_cut including decoherence.
    """

    if t1 == t2:
        if w1 < w_cut or w2 < w_cut:
            return t1, False
        return t1, True
    if t1 > t2:  # make sure t1 < t2
        t1, t2 = t2, t1
        w1, w2 = w2, w1
    # first link has low quality
    if w1 < w_cut:
        return t1, False  # waiting_time = min(t1, t2)
    waiting = np.int(np.floor(t_coh * np.log(w1/w_cut)))
    # first link waits too long
    if t1 + waiting < t2:
        return t1 + waiting, False  # min(t1, t2) < waiting_time < max(t1, t2)
    # second link has low quality
    elif w2 < w_cut:
        return t2, False  # waiting_time = max(t1, t2)
    # both links are good
    else:
        return t2, True  # waiting_time = max(t1, t2)


@nb.jit(nopython=True, error_model="numpy")
def run_time_cut_off(
    t1, t2, w1, w2,
    mt_cut=np.iinfo(np.int).max, w_cut=1.e-8, rt_cut=np.iinfo(np.int).max, t_coh=np.iinfo(np.int).max):
    if t1 > rt_cut or t2 > rt_cut:
        return rt_cut, False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def time_cut_off(
    t1, t2, w1, w2,
    mt_cut=np.iinfo(np.int).max, w_cut=1.e-8, rt_cut=np.iinfo(np.int).max, t_coh=np.iinfo(np.int).max):
    waiting_time1, result1 = memory_cut_off(
        t1, t2, w1, w2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, t_coh=t_coh)
    waiting_time2, result2 = run_time_cut_off(
        t1, t2, w1, w2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, t_coh=t_coh)
    result1 += mt_cut
    result2 += rt_cut
    if result1 and result2:
        return max(waiting_time1, waiting_time2), True
    else:
        # the waiting time of failing cutoff is always
        # smaller than max(t1, t2), so we just need a min here.
        return min(waiting_time1, waiting_time2), False


########################################################################
def join_links_compatible(
        pmf1, pmf2, w_func1, w_func2,
        cutoff=np.iinfo(np.int32).max, ycut=True,
        cut_type="memory_time", evaluate_func=get_one, t_coh=np.inf):
    """
    Calculate P_s and P_f.
    Calculate sum_(t=tA+tB) Pr(TA=tA)*Pr(TB=tB)*f(tA, tB)
    where f is the value function to
    be evaluated for the joint distribution.

    Note
    ----
    For swap the success probability p is
    considered in the iterative convolution.

    For the memory time cut-off,
    the constant shift is added in the iterative convolution.


    Parameters
    ----------
    pmf1, pmf2: array-like
        The waiting time distribution of the two input links, Pr(T=t).
    w_func1, w_func2: array-like
        The Werner parameter function, W(t).
    cutoff: int or float
        The cut-off threshold.
    ycut: bool
        Success ful cut-off or failed cut-off.
    cutoff_type: str
        Type of cut-off.
        `memory_time`, `run_time` or `fidelity`.
    evaluate_func: str
        The function to be evaluated the returns a float number.
        It can be
        ``get_one`` for trival cases\n
        ``get_swap_wout`` for wout\n
        ``get_dist_prob_suc`` for pdist\n
        ``get_dist_prob_fail`` for 1-pdist\n
        ``get_dist_prob_wout`` for pdist * wdist
    t_coh: int or float
        The coherence time of the memory.

    Returns
    -------
    result: array-like 1-D
        The resulting array of joining the two links.
    """
    mt_cut=np.iinfo(np.int32).max
    w_cut=0.0
    rt_cut=np.iinfo(np.int32).max
    if cut_type == "memory_time":
        cutoff_func = memory_cut_off
        mt_cut = cutoff
    elif cut_type == "fidelity":
        cutoff_func = fidelity_cut_off
        w_cut = cutoff
    elif cut_type == "run_time":
        cutoff_func = run_time_cut_off
        rt_cut = cutoff
    else:
        raise NotImplementedError("Unknow cut-off type")

    if evaluate_func == "1":
        evaluate_func = get_one
    elif evaluate_func == "w1w2":
        evaluate_func = get_swap_wout
    elif evaluate_func == "0.5+0.5w1w2":
        evaluate_func = get_dist_prob_suc
    elif evaluate_func == "0.5-0.5w1w2":
        evaluate_func = get_dist_prob_fail
    elif evaluate_func == "w1+w2+4w1w2":
        evaluate_func = get_dist_prob_wout
    elif isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    result = join_links_helper(
        pmf1, pmf2, w_func1, w_func2, cutoff_func=cutoff_func, evaluate_func=evaluate_func, ycut=ycut, t_coh=t_coh, 
        mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut)
    return result


@nb.jit(nopython=True, error_model="numpy")
def join_links_helper(
        pmf1, pmf2, w_func1, w_func2,
        cutoff_func=memory_cut_off, evaluate_func=get_one, ycut=True, t_coh=np.inf, mt_cut=np.iinfo(np.int32).max, w_cut=0.0, rt_cut=np.iinfo(np.int32).max):
    size = len(pmf1)
    result = np.zeros(size, dtype=np.float64)
    decay_factors = np.exp(- np.arange(size) / t_coh)

    for t1 in range(1, size):
        for t2 in range(1, size):
            waiting_time, selection_pass = cutoff_func(
                t1, t2, w_func1[t1], w_func2[t2],
                mt_cut, w_cut, rt_cut, t_coh)
            if not ycut:
                selection_pass = not selection_pass
            if selection_pass:
                result[waiting_time] += pmf1[t1] * pmf2[t2] * \
                    evaluate_func(
                        t1, t2, w_func1[t1], w_func2[t2],
                        decay_factors[np.abs(t1-t2)])
    return result


