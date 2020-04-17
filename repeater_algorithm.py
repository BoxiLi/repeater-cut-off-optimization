"""
This file contains the core algorithm for calculating
the waiting time and the Werner parameter (fidelity)
for repeater chain protocols.
"""
import time
import warnings
from copy import deepcopy
from collections.abc import Iterable
import logging

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
try:
    import cupy as cp
    _use_cupy = True
except (ImportError, ModuleNotFoundError):
    _use_cupy = False

from protocol_units import (
    memory_cut_off,
    get_one, get_swap_wout,
    get_dist_prob_suc, get_dist_prob_fail, get_dist_prob_wout
    )
from utility_functions import secret_key_rate
from logging_utilities import log_init, create_iter_kwargs, save_data
from repeater_mc import repeater_mc, plot_mc_simulation


__all__ = ["compute_unit", "plot_algorithm",
           "join_links", "repeater_sim"]

_GPU_THRESHOLD = 100000
_FFT_CONVOLUTION_THRESHOLD = 1000

@nb.jit(nopython=True, error_model="numpy")
def join_links(
        pmf1, pmf2, w_func1, w_func2,
        mt_cut=np.iinfo(np.int32).max, w_cut=0.0, ycut=True,
        waiting=memory_cut_off, evaluate_fun=get_one, t_coh=np.inf):
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
    value_matrix: array-like 2-D
        The value of a function f(TA, TB) in matrix form.
        f can be the success probability p,
        the function for combining two Werner parameters or
        the product of them.
    mt_cut: int
        The memory time cut-off.
    w_cut: 
        The werner parameter cut-off.
    waiting: jit-compiled function
        Calculate the waiting time for this attempt (with cut-off)
        and if it succeeds.
    evaluate_fun: jit-compiled function
        The function to be evaluated the returns a float number.
        It can be
        ``get_one`` for trival cases\n
        ``get_swap_wout`` for wout\n
        ``get_dist_prob_suc`` for pdist\n
        ``get_dist_prob_fail`` for 1-pdist\n
        ``get_dist_prob_wout`` for pdist * wdist
    t_coh: int or float
        The coherence time of the memory. It is currently not used because
        the decay factor is passed by a global variable.

    Returns
    -------
    result: array-like 1-D
        The resulting array of joining the two links.
    """
    t_trunc = len(pmf1)
    result = np.zeros(t_trunc, dtype=np.float64)
    decay_factors = np.exp(- np.arange(t_trunc) / t_coh)

    for t1 in range(t_trunc):
        for t2 in range(t_trunc):
            waiting_time, selection_pass = waiting(
                t1, t2, w_func1[t1], w_func2[t2],
                mt_cut, w_cut, t_coh, ycut)
            if not ycut:
                selection_pass = not selection_pass
            # if abs(t1 - t2) > mt_cut and ycut:
            #     continue
            # if cut_func(t1, t2) and not ycut:
            #     continue
            if selection_pass:
                result[waiting_time] += pmf1[t1] * pmf2[t2] * \
                    evaluate_fun(
                        t1, t2, w_func1[t1], w_func2[t2],
                        decay_factors[np.abs(t1-t2)])
    return result


def iterative_convolution(
        func, t_trunc, shift=0, first_func=None, coeffs=None):
    """
    Calculate the convolution iteratively:
    first_func * func * func * ... * func
    It the return the sum of all iterative convolution:
    first_func + first_func * func + first_func * func * func ...

    Parameters
    ----------
    func: array-like
        The function to be convolved in array form.
    t_trunc: int
        The truncation time that determines the number of sums
    shift: int, optional
        For each k the function will be shifted to the right. Using for
        time-out mt_cut.
    first_func: array-like, optional
        The first_function in the convolution. If not given, use func.
        It can be different because the first_func is `P_s` and the `func` P_f.
    coeffs: array-like optional
        The additional factor when sum over k, default is 1.

    Returns
    -------
    sum_convolved: array-like
        The result of the sum of all convolutions.
    """
    if first_func is None:
        first_func = func
    if coeffs is None:
        # constant swap success probability p
        coeffs = np.ones(t_trunc)
    convolved = first_func
    if shift != 0:
        # mt_cut is added here.
        # because it is a constant, we only need t_trunc/mt_cut convolution.
        max_k = int(np.ceil((t_trunc/shift)))
    else:
        max_k = t_trunc
    sum_convolved = coeffs[0] * convolved

    # decide what convolution to use and prepare the data
    length = len(first_func)
    if length != len(func):
        raise ValueError(
            "Input for the convolution must have the same length.")
    if length > _FFT_CONVOLUTION_THRESHOLD:
        shape = length + length - 1
        fsize = 2 ** np.ceil(np.log2(shape)).astype(int)
        if _use_cupy and length > _GPU_THRESHOLD:
            # transfer the data to GPU
            coeffs = cp.asarray(coeffs)
            sum_convolved = cp.asarray(sum_convolved)
            convolved = cp.asarray(convolved)
            func = cp.asarray(func)
            # use CuPy fft
            ifft = cp.fft.ifft
            fft = cp.fft.fft
            to_real = cp.real
        else:
            # use NumPy fft
            ifft = np.fft.ifft
            fft = np.fft.fft
            to_real = np.real
        convolved_fourier = fft(convolved, fsize)
        func_fourier = fft(func, fsize)

    # perform convolution
    for k in range(1, max_k):
        if length > _FFT_CONVOLUTION_THRESHOLD:  # convolution in the fourier space
            convolved_fourier = convolved_fourier * func_fourier
            convolved = ifft(convolved_fourier)[:length]
            convolved = to_real(convolved)
        else:
            convolved = np.convolve(convolved, func)[:length]
        # The first k+1 elements should be 0, but FFT convolution
        # gives a non-zero value of about, e-20. It remains to
        # see wether this will have effect on other elements
        convolved[:k+1] = 0.
        sum_convolved[k*shift:] += coeffs[k] * convolved[:t_trunc-k*shift]
    if _use_cupy and length > _GPU_THRESHOLD:
        sum_convolved = cp.asnumpy(sum_convolved)
    return sum_convolved


def entanglement_swap(
        pmf1, w_func1, pmf2, w_func2, p_swap,
        mt_cut, w_cut, t_coh, t_trunc, cut_type):
    """
    Calculate the waiting time and average Werner parameter with time-out
    for entanglement swap.

    Parameters
    ----------
    pmf1, pmf2: array-like 1-D
        The waiting time distribution of the two input links.
    w_func1, w_func2: array-like 1-D
        The Werner parameter as function of T of the two input links.
    p_swap: float
        The success probability of entanglement swap.
    mt_cut: int
        The memory time cut-off.
    w_cut: 
        The werner parameter cut-off.
    t_coh: int
        The coherence time.
    t_trunc: int
        The truncation time, also the size of the matrix.

    Returns
    -------
    t_pmf: array-like 1-D
        The waiting time distribution of the entanglement swap.
    w_func: array-like 1-D
        The Werner parameter as function of T of the entanglement swap.
    """
    if cut_type == "memory_time":
        fail_attempt_waiting = memory_cut_off
        suc_attempt_waiting = memory_cut_off
        shift = mt_cut
    elif cut_type == "run_time":
        raise NotImplementedError
    elif cut_type == "fidelity":
        raise NotImplementedError
    else:
        raise ValueError("Unknow cut-off type")
    goem_coeff = p_swap*(1-p_swap)**(np.arange(t_trunc))

    # P'_f
    cut_off_attempt_fail_pmf = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=False,
        mt_cut=mt_cut, w_cut=w_cut, waiting=fail_attempt_waiting, t_coh=t_coh)
    # P'_s
    cut_off_attempt_suc_pmf = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=True,
        mt_cut=mt_cut, w_cut=w_cut, waiting=suc_attempt_waiting, t_coh=t_coh)
    # P_f or P_s (Differs only by a constant p_swap)
    cut_off_pmf = iterative_convolution(
        cut_off_attempt_fail_pmf, t_trunc, shift=shift,
        first_func=cut_off_attempt_suc_pmf)
    # Pr(Tout = t)
    swap_pmf = iterative_convolution(
        cut_off_pmf, t_trunc, shift=0, coeffs=goem_coeff)

    # Wsuc * P_s
    cut_off_attempt_w = join_links(
        pmf1, pmf2, w_func1=w_func1, w_func2=w_func2, ycut=True,
        mt_cut=mt_cut, w_cut=w_cut, waiting=suc_attempt_waiting,
        t_coh=t_coh, evaluate_fun=get_swap_wout)
    # Wprep * Pr(Tout = t)
    cut_off_w = iterative_convolution(
        cut_off_attempt_fail_pmf, t_trunc,
        shift=shift, first_func=cut_off_attempt_w)
    # Wout * Pr(Tout = t)
    swap_w_func = iterative_convolution(
        cut_off_pmf, t_trunc, shift=0,
        first_func=cut_off_w, coeffs=goem_coeff)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swap_w_func[:] /= swap_pmf[0:]  # 0-th element has 0 pmf
    return swap_pmf, swap_w_func


def destillation(
        pmf1, w_func1, pmf2, w_func2,
        mt_cut, w_cut, t_coh, t_trunc, cut_type):
    """
    Calculate the waiting time and average Werner parameter
    with time-out for the distillation.

    Parameters
    ----------
    pmf1, pmf2: array-like 1-D
        The waiting time distribution of the two input links.
    w_func1, w_func2: array-like 1-D
        The Werner parameter as function of T of the two input links.
    mt_cut: int
        The memory time cut-off.
    w_cut: 
        The werner parameter cut-off.
    t_coh: int
        The coherence time.
    t_trunc: int
        The truncation time, also the size of the matrix.

    Returns
    -------
    t_pmf: array-like 1-D
        The waiting time distribution of the distillation.
    w_func: array-like 1-D
        The Werner parameter as function of T of the distillation.
    """
    if cut_type == "memory_time":
        fail_attempt_waiting = memory_cut_off
        suc_attempt_waiting = memory_cut_off
        shift = mt_cut
    elif cut_type == "run_time":
        raise NotImplementedError
    elif cut_type == "fidelity":
        raise NotImplementedError
    else:
        raise ValueError("Unknow cut-off type")
    # P'_f
    cut_off_attempt_fail_w = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=False,
        mt_cut=mt_cut, w_cut=w_cut, waiting=fail_attempt_waiting,
        t_coh=t_coh)
    # P'_ss
    cut_off_attempt_suc_dist_suc_pmf = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=True,
        mt_cut=mt_cut, w_cut=w_cut, waiting=suc_attempt_waiting,
        evaluate_fun=get_dist_prob_suc, t_coh=t_coh)
    # P'_sf
    dist_fail_time_out_suc_pmf = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=True,
        mt_cut=mt_cut, w_cut=w_cut, waiting=suc_attempt_waiting,
        evaluate_fun=get_dist_prob_fail, t_coh=t_coh)
    # P_s
    time_out_dist_suc_pmf = iterative_convolution(
        cut_off_attempt_fail_w, t_trunc, shift=shift,
        first_func=cut_off_attempt_suc_dist_suc_pmf)
    # P_f
    time_out_dist_fail_pmf = iterative_convolution(
        cut_off_attempt_fail_w, t_trunc, shift=shift,
        first_func=dist_fail_time_out_suc_pmf)
    # Pr(Tout = t)
    dist_pmf = iterative_convolution(
        time_out_dist_fail_pmf, t_trunc, shift=0,
        first_func=time_out_dist_suc_pmf)

    # Wsuc * P'_ss
    cut_off_attempt_w = join_links(
        pmf1, pmf2, w_func1, w_func2, ycut=True,
        mt_cut=mt_cut, w_cut=w_cut, waiting=suc_attempt_waiting,
        evaluate_fun=get_dist_prob_wout, t_coh=t_coh)
    # Wprep * P_s
    cut_off_w = iterative_convolution(
        cut_off_attempt_fail_w, t_trunc, shift=shift,
        first_func=cut_off_attempt_w)
    # Wout * Pr(Tout = t)
    dist_w_func = iterative_convolution(
        time_out_dist_fail_pmf, t_trunc, shift=0,
        first_func=cut_off_w)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist_w_func[1:] /= dist_pmf[1:]
    return dist_pmf, dist_w_func


def compute_unit(
        parameters, pmf1, w_func1, pmf2=None, w_func2=None, unit_kind="swap"):
    """
    Calculate the the waiting time distribution and
    the Werner parameter of a protocol unit swap or distillation.
    Cut-off is built in swap or distillation.

    Parameters
    ----------
    parameters: dict
        A dictionary contains the parameters of
        the repeater and the simulation.
    pmf1, pmf2: array-like 1-D
        The waiting time distribution of the two input links.
    w_func1, w_func2: array-like 1-D
        The Werner parameter as function of T of the two input links.
    unit_kind: str
        "swap" or "dist"

    Returns
    -------
    t_pmf, w_func: array-like 1-D
        The output waiting time and Werner parameters
    """
    if pmf2 is None:
        pmf2 = pmf1
    if w_func2 is None:
        w_func2 = w_func1
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    mt_cut = parameters["tau"]
    t_coh = parameters["t_coh"]
    w0 = parameters["w0"]
    t_trunc = parameters["t_trunc"]
    if "cut_type" not in parameters:
        cut_type = "memory_time"
    else:
        cut_type = parameters["cut_type"]
    if "w_cut" not in parameters:
        w_cut = 0.0
    else:
        w_cut = parameters["w_cut"]
    if not isinstance(p_gen, float) or not isinstance(p_swap, float):
        raise TypeError("p_gen and p_swap must be a float number.")
    if not np.issubdtype(type(mt_cut), np.integer):
        raise TypeError(f"Memory cut-off must be an integer. not {mt_cut}")
    if not np.isreal(t_coh):
        raise TypeError(
            f"The coherence time must be a real number, not{t_coh}")
    if not np.isreal(w0) or w0 < 0. or w0 > 1.:
        raise TypeError(f"Invalid Werner parameter w0 = {w0}")

    # swap or distillation for next level
    if unit_kind == "swap":
        pmf, w_func = entanglement_swap(
            pmf1, w_func1, pmf2, w_func2, p_swap,
            mt_cut, w_cut, t_coh, t_trunc, cut_type=cut_type)
    elif unit_kind == "dist":
        pmf, w_func = destillation(
            pmf1, w_func1, pmf2, w_func2,
            mt_cut, w_cut, t_coh, t_trunc, cut_type=cut_type)

    # erase ridiculous Werner parameters,
    # it can happen when the probability is too small ~1.0e-20.
    w_func = np.where(np.isnan(w_func), 1., w_func)
    w_func[w_func > 1.0] = 1.0
    w_func[w_func < 0.] = 0.

    # check coverage
    coverage = np.sum(pmf)
    if coverage < 0.99:
        logging.warning(
            "The truncation time only covers {:.2f}% of the distribution, "
            "please increase t_trunc.\n".format(
                coverage*100))

    return pmf, w_func


def repeater_sim(parameters, all_level=False):
    """
    Compute the waiting time and the Werner parameter of a symmetric
    repeater protocol.

    Parameters
    ----------
    parameters: dict
        A dictionary contains the parameters of
        the repeater and the simulation.
    all_level: bool
        If true, Return a list of the result of all the levels.
        [(t_pmf0, w_func0), (t_pmf1, w_func1) ...]

    Returns
    -------
    t_pmf, w_func: array-like 1-D
        The output waiting time and Werner parameters
    """
    parameters = deepcopy(parameters)
    protocol = parameters["protocol"]
    p_gen = parameters["p_gen"]
    mt_cut = parameters["tau"]
    if not isinstance(mt_cut, Iterable):
        mt_cut = (mt_cut,) * len(protocol)
    else:
        mt_cut = tuple(mt_cut)
    w0 = parameters["w0"]
    t_trunc = parameters["t_trunc"]

    # elementary link
    t_list = np.arange(1, t_trunc)
    pmf = p_gen * (1 - p_gen)**(t_list - 1)
    pmf = np.concatenate((np.array([0.]), pmf))
    w_func = np.array([w0] * t_trunc)
    if all_level:
        full_result = [(pmf, w_func)]

    # protocol unit level by level
    for i, operation in enumerate(protocol):
        parameters["tau"] = mt_cut[i]
        if operation == 0:
            pmf, w_func = compute_unit(
                parameters, pmf, w_func, unit_kind="swap")
        elif operation == 1:
            pmf, w_func = compute_unit(
                parameters, pmf, w_func, unit_kind="dist")
        if all_level:
            full_result.append((pmf, w_func))

    if all_level:
        return full_result
    else:
        return pmf, w_func


def plot_algorithm(pmf, w_func, axs=None, t_trunc=None, legend=None):
    """
    Plot the waiting time distribution and Werner parameters
    """
    cdf = np.cumsum(pmf)
    if t_trunc is None:
        try:
            t_trunc = np.min(np.where(cdf >= 0.997))
        except ValueError:
            t_trunc = len(pmf)
    pmf = pmf[:t_trunc]
    w_func = w_func[:t_trunc]
    w_func[0] = np.nan

    axs[0][0].plot((np.arange(t_trunc)), np.cumsum(pmf))

    axs[0][1].plot((np.arange(t_trunc)), pmf)
    axs[0][1].set_xlabel("Waiting time $T$")
    axs[0][1].set_ylabel("Probability")

    axs[1][0].plot((np.arange(t_trunc)), w_func)
    axs[1][0].set_xlabel("Waiting time $T$")
    axs[1][0].set_ylabel("Werner parameter")

    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
