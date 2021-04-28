import logging

import numpy as np
import numba as nb


__all__ = ["join_links_efficient"]


##############################################################################
"""
Swap and distillation for werner parameter

Parameters
----------
pmf1, pmf2: 1d array-like
    The waiting time distribution of the two input links, Pr(T=t).
w_func1, w_func2: 1d array-like
    The Werner parameter function, W(t).
t_coh: int or float
    The coherence time of the memory.

Returns
-------
Four arrays of the shape `t_trunc` used in efficient computation.
"""
def get_one_werner(pmf1, pmf2, w_func1, w_func2, t_coh):
    return pmf1, pmf1, pmf2, pmf2


def get_w1w2_werner(pmf1, pmf2, w_func1, w_func2, t_coh):
    """
    func1_later = Pr(T1=t1) * w1 * exp(-t1)
    func1_early = Pr(T1=t1) * w1 * exp(+t1)
    func2_later = Pr(T2=t2) * w2 * exp(-t2)
    func1_early = Pr(T2=t2) * w2 * exp(+t2)
    """
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh) # exp(-t)
    func1_later = pmf1 * w_func1 * decay_factors
    func1_early = pmf1 * w_func1 / decay_factors
    func2_later = pmf2 * w_func2 * decay_factors
    func2_early = pmf2 * w_func2 / decay_factors
    return func1_early, func1_later, func2_early, func2_later


def get_w1_werner(pmf1, pmf2, w_func1, w_func2, t_coh):
    """
    func1_later = Pr(T1=t1) * w1 * exp(-t1)
    func1_early = Pr(T1=t1) * w1 * exp(+t1)
    func2_later = Pr(T2=t2) * exp(-t2)
    func1_early = Pr(T2=t2) * exp(+t2)
    """
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    func1_later = pmf1 * w_func1
    func2_early = pmf2
    func1_early = pmf1 * w_func1 / decay_factors
    func2_later = pmf2 * decay_factors
    return func1_early, func1_later, func2_early, func2_later


def get_w2_werner(pmf1, pmf2, w_func1, w_func2, t_coh):
    """
    func1_later = Pr(T1=t1) * exp(-t1)
    func1_early = Pr(T1=t1) * exp(+t1)
    func2_later = Pr(T2=t2) * w2 * exp(-t2)
    func1_early = Pr(T2=t2) * w2 * exp(+t2)
    """
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    func1_later = pmf1 * decay_factors
    func2_early = pmf2 * w_func2 / decay_factors
    func1_early = pmf1
    func2_later = pmf2 * w_func2
    return func1_early, func1_later, func2_early, func2_later


##############################################################################
"""
State representation for entanglement swapping
The post swapping state is given by
m2 * (n1 + n2) + m1 * (n3 + n4), where * means element-wise multiplication.
See the Mathematica notebook for detailed calculation.

Parameters
----------
pmf1, pmf2: 1d array-like
    The waiting time distribution of the two input links, Pr(T=t).
m, n: 3d array-like
    The time dependent array of quantum states
    with the shape `[t_trunc,4,4]`.
t_coh: int or float
    The coherence time of the memory.

Returns
-------
Four arrays of the shape `[t_trunc,4,4]` used in efficient computation.
"""
@nb.jit(nopython=True, error_model="python")
def identity_matrix_array(shape, dtype):
    identity_array = np.zeros(shape, dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            identity_array[i, j, j] = 1.
    return identity_array


def get_decay_dm(pmf1, pmf2, m, n, t_coh):
    """)
    func1_later = Pr(T1=t1) * I/4 * exp(-t1)
    func1_early = Pr(T1=t1) * I/4 * exp(+t1)
    func2_later = Pr(T2=t2) * I/4 * exp(-t2)
    func1_early = Pr(T2=t2) * I/4 * exp(+t2)
    """
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    identity = identity_matrix_array(m.shape, m.dtype)
    identity = np.transpose(identity, (1, 2, 0))
    func1_later = 1/4. * pmf1 * identity * decay_factors
    func1_early = 1/4. * pmf1 * identity / decay_factors
    func2_later = pmf2 * identity * decay_factors
    func2_early = pmf2 * identity / decay_factors
    func1_later = np.transpose(func1_later, (2, 0, 1))
    func1_early = np.transpose(func1_early, (2, 0, 1))
    func2_later = np.transpose(func2_later, (2, 0, 1))
    func2_early = np.transpose(func2_early, (2, 0, 1))
    return func1_early, func1_later, func2_early, func2_later


def get_one_dm(pmf1, pmf2, m, n, t_coh):
    """
    func1 = Pr(T1=t1) * I/4
    func2 = Pr(T2=t2) * I/4
    """
    identity = np.zeros(m.shape, m.dtype)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            identity[i, j, j] = 1.
    identity = np.transpose(identity, (1, 2, 0))
    func1_later = pmf1 * identity * 1/4.
    func1_early = pmf1 * identity * 1/4.
    func2_later = pmf2 * identity
    func2_early = pmf2 * identity
    func1_later = np.transpose(func1_later, (2, 0, 1))
    func1_early = np.transpose(func1_early, (2, 0, 1))
    func2_later = np.transpose(func2_later, (2, 0, 1))
    func2_early = np.transpose(func2_early, (2, 0, 1))
    return func1_early, func1_later, func2_early, func2_later


def apply_to_all_time_wrapper(func):
    """
    Broad cast the given function acting on
    an array with the shape `[4, 4]`
    to a function acting on an array with the shape `[t_trunc, 4, 4]`.
    """
    @nb.jit(nopython=True, error_model="python")
    def inner(state_array):
        result = np.empty(state_array.shape, dtype=state_array.dtype)
        for i in range(state_array.shape[0]):
            result[i] = func(state_array[i])
        return result
    return inner


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def m1(m):
    return np.asarray([
        [m[1, 1], m[1, 0], m[1, 3], m[1, 2]],
        [m[0, 1], m[0, 0], m[0, 3], m[0, 2]],
        [m[3, 1], m[3, 0], m[3, 3], m[3, 2]],
        [m[2, 1], m[2, 0], m[2, 3], m[2, 2]]
        ])


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def m2(m):
    return np.asarray([
        [m[0, 0], m[0, 1], m[0, 2], m[0, 3]],
        [m[1, 0], m[1, 1], m[1, 2], m[1, 3]],
        [m[2, 0], m[2, 1], m[2, 2], m[2, 3]],
        [m[3, 0], m[3, 1], m[3, 2], m[3, 3]]
        ])


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def n1(n):
    return np.asarray([
        [n[1, 1], n[1, 2], n[1, 1], n[1, 2]],
        [n[2, 1], n[2, 2], n[2, 1], n[2, 2]],
        [n[1, 1], n[1, 2], n[1, 1], n[1, 2]],
        [n[2, 1], n[2, 2], n[2, 1], n[2, 2]]
        ])


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def n2(n):
    return np.asarray([
        [n[2, 2], n[2, 1], n[2, 2], n[2, 1]],
        [n[1, 2], n[1, 1], n[1, 2], n[1, 1]],
        [n[2, 2], n[2, 1], n[2, 2], n[2, 1]],
        [n[1, 2], n[1, 1], n[1, 2], n[1, 1]]
        ])


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def n3(n):
    return np.asarray([
        [n[0, 0], n[0, 3], n[0, 0], n[0, 3]],
        [n[3, 0], n[3, 3], n[3, 0], n[3, 3]],
        [n[0, 0], n[0, 3], n[0, 0], n[0, 3]],
        [n[3, 0], n[3, 3], n[3, 0], n[3, 3]]
        ])


@apply_to_all_time_wrapper
@nb.jit(nopython=True, error_model="python")
def n4(n):
    return np.asarray([
        [n[3, 3], n[3, 0], n[3, 3], n[3, 0]],
        [n[0, 3], n[0, 0], n[0, 3], n[0, 0]],
        [n[3, 3], n[3, 0], n[3, 3], n[3, 0]],
        [n[0, 3], n[0, 0], n[0, 3], n[0, 0]]
        ])


def get_mn_array(pmf1, pmf2, m, n, t_coh):
    """
    Compute
        pA M(tA) exp(-tA/t_coh)
        pA M(tA) exp(+tA/t_coh)
        pB N(tB) exp(-tB/t_coh)
        pB N(tB) exp(+tB/t_coh)
    They will be used to compute e.g.
        pA * pB * M(tA) * N(tB) exp((tA-tB)/t_coh)
    """
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    m = np.transpose(m, (1, 2, 0))
    n = np.transpose(n, (1, 2, 0))
    func1_later = pmf1 * m * decay_factors
    func1_early = pmf1 * m / decay_factors
    func2_later = pmf2 * n * decay_factors
    func2_early = pmf2 * n / decay_factors
    func1_later = np.transpose(func1_later, (2, 0, 1))
    func1_early = np.transpose(func1_early, (2, 0, 1))
    func2_later = np.transpose(func2_later, (2, 0, 1))
    func2_early = np.transpose(func2_early, (2, 0, 1))
    return func1_early, func1_later, func2_early, func2_later


def get_m2n1_array(pmf1, pmf2, m, n, t_coh):
    return get_mn_array(pmf1, pmf2, m2(m), n1(n), t_coh)


def get_m2n2_array(pmf1, pmf2, m, n, t_coh):
    return get_mn_array(pmf1, pmf2, m2(m), n2(n), t_coh)


def get_m1n3_array(pmf1, pmf2, m, n, t_coh):
    return get_mn_array(pmf1, pmf2, m1(m), n3(n), t_coh)


def get_m1n4_array(pmf1, pmf2, m, n, t_coh):
    return get_mn_array(pmf1, pmf2, m1(m), n4(n), t_coh)


###############################################################################
# API function for merging two entangled states by swap or distillation.

def join_links_efficient(
        pmf1, pmf2, w_func1, w_func2,
        cutoff=np.iinfo(np.int32).max, ycut=True,
        cut_type=None, evaluate_func=None, t_coh=np.inf):
    """
    Calculate P_s and P_f efficiently using cumulative function.
    Only memory time cutoff is supported.
    The implementation includes both werner parameter representation
    and density matrix representation.

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
        The function used to evaluate the distribution.
    t_coh: int or float
        The coherence time of the memory.

    Returns
    -------
    result: array-like 1-D
        The resulting array of joining the two links.
    """
    if cut_type == "memory_time":
        mt_cut = cutoff
    else:
        raise NotImplementedError("Unknown cut-off type.")

    if evaluate_func == "1":
        evaluate_coeff_list = (1,)
        final_coeff = 1.
        evaluate_func_list = (get_one_werner,)
        kind = "probability"
    elif evaluate_func == "w1w2":
        if len(w_func1.shape) == 1:
            evaluate_coeff_list = (1,)
            final_coeff = 1.
            evaluate_func_list = (get_w1w2_werner,)
        else:
            # m2 * (n1 + n2) + m1 * (n3 + n4), ELEMENT-WISE
            evaluate_coeff_list = (
            1., 
            -1.,
            1., 
            1., 
            1., 
            1., 
            )
            final_coeff = 1.
            evaluate_func_list = (
                get_one_dm,
                get_decay_dm,
                get_m2n1_array,
                get_m2n2_array,
                get_m1n3_array,
                get_m1n4_array,
                )
        kind = "state"
    elif evaluate_func == "0.5+0.5w1w2":
        evaluate_coeff_list = (0.5, 0.5)
        final_coeff = 1.
        evaluate_func_list = (get_one_werner, get_w1w2_werner)
        kind = "probability"
    elif evaluate_func == "0.5-0.5w1w2":
        evaluate_coeff_list = (0.5, -0.5)
        final_coeff = 1.
        evaluate_func_list = (get_one_werner, get_w1w2_werner)
        kind = "probability"
    elif evaluate_func == "w1+w2+4w1w2":
        evaluate_coeff_list = (1., 1., 4.)
        final_coeff = 1./6.
        evaluate_func_list = (get_w1_werner, get_w2_werner, get_w1w2_werner)
        kind = "state"
    elif isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    
    size = len(pmf1)
    if size/t_coh > 300:
        logging.warn("Overflow in the exponential function!")
    if kind == "probability":
        final_result = np.zeros(pmf1.shape, dtype=pmf1.dtype)
    elif kind == "state":
        final_result = np.zeros(w_func1.shape, dtype=w_func1.dtype)
    for evaluate_coeff, evaluate_func in zip(evaluate_coeff_list, evaluate_func_list):
        # separate the positive and negative part for numerical stability

        if kind == "probability":
            result = np.zeros(pmf1.shape, dtype=pmf1.dtype)
            minus_result = np.zeros(pmf1.shape, dtype=pmf1.dtype)   
        elif kind == "state":
            result = np.zeros(w_func1.shape, dtype=w_func1.dtype)
            minus_result = np.zeros(w_func1.shape, dtype=w_func1.dtype)
        func1_early, func1_later, func2_early, func2_later = evaluate_func(
            pmf1, pmf2, w_func1, w_func2, t_coh)

        if ycut:
            cum_func1_early = cumsum(func1_early)
            cum_func2_early = cumsum(func2_early)
            result = join_with_suc_cutoff(
                cutoff, result, minus_result,
                cum_func1_early, func1_later, cum_func2_early, func2_later)
        else:
            # waiting time is min(t1, t2), fix the early link as t
            cum_func1_later = cumsum(func1_later)
            cum_func2_later = cumsum(func2_later)
            result = join_with_fail_cutoff(
                cutoff, result, minus_result,
                func1_early, func1_later, cum_func1_later,
                func2_early, func2_later, cum_func2_later)
        final_result += evaluate_coeff * result
    final_result = final_coeff * final_result
    return final_result


@nb.jit(nopython=True, error_model="python")
def join_with_suc_cutoff(
        cutoff, result, minus_result, cum_func1_early, func1_later, cum_func2_early, func2_later):
    """
    Core algorithms for efficient computation of
    the time and state distribution.
    """
    for t in range(1, len(result)):
        cut = max(0, t - cutoff - 1)
        # link 2 wait
        result[t] += func1_later[t] * cum_func2_early[t]
        minus_result[t] += func1_later[t] * cum_func2_early[cut]
        # link 1 wait
        result[t] += func2_later[t] * cum_func1_early[t-1]
        minus_result[t] += func2_later[t] * cum_func1_early[cut]
    result -= minus_result
    return result


@nb.jit(nopython=True, error_model="python")
def join_with_fail_cutoff(
        cutoff, result, minus_result,
        func1_early, func1_later, cum_func1_later, func2_early, func2_later, cum_func2_later):
    """
    waiting time is min(t1, t2), fix the early link as t
    """
    for t in range(1, len(result) - cutoff):
        cut = t + cutoff
        # link 1 wait
        result[t] += func2_early[t] * cum_func1_later[-1]
        minus_result[t] += func2_later[t] * cum_func1_later[cut]
        # link 2 wait
        result[t] += func1_early[t] * cum_func2_later[-1]
        minus_result[t] += func1_later[t] * cum_func2_later[cut]
    result -= minus_result
    return result


def cumsum(array):
    if len(array.shape) == 1:
        return np.cumsum(array)
    else:
        return state_cumsum(array)


@nb.jit(nopython=True, error_model="python")
def state_cumsum(array):
    for i in range(0, array.shape[0]-1):
        array[i+1] += array[i]
    return array
