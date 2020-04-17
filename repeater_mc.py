"""
This file contains the Monte Carlo's algorithm used in the validation
of the repeater simulation.
"""
import time
from collections.abc import Iterable

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


__all__ = ["repeater_mc", "plot_mc_simulation"]


def repeater_mc(parameters, multiprocessing=False, return_pmf=False):
    """
    Main function for the Monte Carlo simulation. The parameters are
    unpacked and the sampled data are post-processed.
    """
    # unpack parameters
    protocol = parameters["protocol"]
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    w0 = parameters["w0"]
    tau = parameters["tau"]
    if not isinstance(tau, Iterable):
        tau = np.array([tau]*len(protocol))
    tau = np.array(tau)
    t_coh = parameters["t_coh"]
    sample_size = parameters["sample_size"]
    network_parameters = (
        protocol, p_gen, p_swap, tau, w0, t_coh)

    # run simulation
    t_samples_list, w_samples_list = mc_sample(
        network_parameters, sample_size, multiprocessing)
    if not return_pmf:
        return t_samples_list, w_samples_list

    # samples to distribution
    t_trunc = int(np.ceil(np.max(t_samples_list) * 2 / 3))
    pmf, bin_edges = create_pmf_from_samples(
        t_samples_list, t_trunc, bin_width=1)
    pmf = np.concatenate([np.zeros(bin_edges[0]), pmf])
    w_func = compute_werner(
        t_samples_list, w_samples_list, bin_edges, t_trunc)
    w_func = np.concatenate(
        [[np.nan for i in range(bin_edges[0])], w_func])
    return pmf, w_func


@nb.jit(nopython=True)
def mc_sample(network_parameters, sample_size, multiprocessing=True):
    """
    Perform the Monte Carlo simulation.

    Parameters
    ----------
    network_parameters: tuple
        Parameters in tuple form.
    swap_level: int
        The swap_level of entanglement swap.
    sample_size: int
        The number of sampling.
    multiprocessing: boolean
        If multiprocessing is used.

    Returns
    -------
    t_samples_list: list
        Samples of the waiting time.
    w_samples_list: list
        Samples of the Werner parameters.
    """
    protocol, p_gen, p_swap, tau, w0, t_coh = network_parameters

    t_samples_list = np.empty(sample_size, dtype=np.int64)
    w_samples_list = np.empty(sample_size, dtype=np.float64)

    for k in range(sample_size):
        (time, w) = sample_protocol(
            len(protocol)-1, network_parameters=network_parameters)
        t_samples_list[k] = time
        w_samples_list[k] = w

    return t_samples_list, w_samples_list


@nb.jit(nopython=True)
def sample_protocol(step, network_parameters):
    protocol, p_gen, p_swap, tau, w0, t_coh = \
        network_parameters
    if step == -1:
        time = np.random.geometric(p_gen)
        return (time, w0)
    if protocol[step] == 0:
        t, w = sample_swap(step, network_parameters)
    elif protocol[step] == 1:
        t, w = sample_dist(step, network_parameters)
    return t, w


@nb.jit(nopython=True)
def time_out_both_swap(step, network_parameters):
    """
    Simulate the time out protocol that discards both link when waiting time
    difference is larger than the cut-off tau. Both qubits are discarded when
    it fails.
    """
    protocol, p_gen, p_swap, tau, w0, t_coh = network_parameters
    tau = tau[step]

    tA, wA = sample_protocol(step-1, network_parameters)
    tB, wB = sample_protocol(step-1, network_parameters)
    t_tot = min(tA, tB)
    while np.abs(tA - tB) > tau:
        tA, wA = sample_protocol(step-1, network_parameters)
        tB, wB = sample_protocol(step-1, network_parameters)
        t_tot += tau + min(tA, tB)
    t_tot += abs(tA - tB)
    return (t_tot, tA, tB, wA, wB)


@nb.jit(nopython=True)
def time_out_both_dist(step, network_parameters):
    """
    This should be keep identical to time_out_both_swap.
    This duplication exists
    because numba is not stable with complicated recursive behavior.
    Both qubits are discarded when it fails.
    """
    protocol, p_gen, p_swap, tau, w0, t_coh = \
        network_parameters
    tau = tau[step]

    tA, wA = sample_protocol(step-1, network_parameters)
    tB, wB = sample_protocol(step-1, network_parameters)
    t_tot = min(tA, tB)
    while np.abs(tA - tB) > tau:
        tA, wA = sample_protocol(step-1, network_parameters)
        tB, wB = sample_protocol(step-1, network_parameters)
        t_tot += tau + min(tA, tB)
    t_tot += abs(tA - tB)
    return (t_tot, tA, tB, wA, wB)


@nb.jit(nopython=True)
def sample_swap(step, network_parameters):
    """
    Simulate the entanglement swap recursively with cut-off time.
    """
    protocol, p_gen, p_swap, tau, w0, t_coh = network_parameters

    t_tot, tA, tB, wA, wB = time_out_both_swap(step, network_parameters)
    w = wA * wB * np.exp(-np.abs(tA - tB)/t_coh)

    swap_success = np.random.random()
    if(swap_success <= p_swap):
        return (t_tot, w)
    else:
        time_retry, w_retry = sample_swap(step, network_parameters)
        return (t_tot + time_retry, w_retry)


@nb.jit(nopython=True)
def sample_dist(step, network_parameters):
    """
    Simulate the distillation recursively with cut-off time
    """
    protocol, p_gen, p_swap, tau, w0, t_coh = network_parameters

    t_tot, tA, tB, wA, wB = time_out_both_dist(step, network_parameters)

    if tA < tB:
        wA *= np.exp(-(tB-tA)/t_coh)
    else:
        wB *= np.exp(-(tA-tB)/t_coh)
    p_dist = (1. + wA * wB)/2.
    w = (wA + wB + 4 * wA * wB) / 6. / p_dist

    dist_success = np.random.random()
    if dist_success <= p_dist:
        return (t_tot, w)
    else:
        t_retry, w_retry = sample_dist(step, network_parameters)
        return (t_tot + t_retry, w_retry)


def create_pmf_from_samples(
        t_samples_list, t_trunc=None, bin_width=None, num_bins=None):
    """
    Compute the probability distribution of the waiting time from the sampled data.

    Parameters
    ----------
    t_samples_list : array-like 1-D
        Samples of the waiting time.
    t_trunc: int
        The truncation time.
    bin_width: int
        The width of the bins for the histogram.
    num_binms: int
        The number of bins for the histogram.
        If num_bins and bin_width are both given, bin_width has priority.

    Returns
    -------
    pmf: array-like 1-D
        The probability distribution also
        the normalized histogram of waiting time.
    bin_edges: array-like 1-D
        The edge of each bins from ``numpy.histogram``.
    """
    if t_trunc is None:
        t_trunc = max(t_samples_list)
    if bin_width is None:
        if num_bins is None:
            bin_width = int(np.ceil(t_trunc/200))
        else:
            bin_width = int(np.ceil(t_trunc/num_bins))
    start = np.min(t_samples_list)
    pmf, bin_edges = np.histogram(
        t_samples_list, bins=np.arange(start, t_trunc+1, bin_width))
    return pmf/len(t_samples_list), bin_edges


@nb.jit
def compute_werner(t_samples_list, w_samples_list, bin_edges, t_trunc):
    """
    Compute the average Werner parameters from the sampled data.

    Parameters
    ----------
    t_samples_list : array-like 1-D
        Samples of the waiting time.
    w_samples_list: array-like 1-D
        Samples of the Werner parameters.
    bin_width: int
        The width of the bins for the histogram.
    t_trunc: int
        The truncation time.

    Returns
    -------
    w_func: array-like 1-D
        The averaged Werner parameter as function of T in array form.
    """
    delta_t = bin_edges[1] - bin_edges[0]
    num_bins = len(bin_edges) - 1
    counter = np.zeros(num_bins)
    w_sum = np.zeros(num_bins)
    for k, t in enumerate(t_samples_list):
        if t < t_trunc:
            pos = int((t - bin_edges[0]) / delta_t)
            w_sum[pos] += w_samples_list[k]
            counter[pos] += 1
    return w_sum / counter


def plot_mc_simulation(
        samples_data, axs, num_bins=None, bin_width=None,
        t_trunc=None, legend=None, parameters=None):
    """
    Plot the sampled data from Monte Carlo algorithm
    """
    t_samples_all = samples_data[0]
    if t_trunc is None:
        t_trunc = int(np.ceil(np.max(np.concatenate(t_samples_all)) * 2 / 3))
    for i in range(len(t_samples_all)):
        pmf, bin_edges = create_pmf_from_samples(
            t_samples_all[i], t_trunc=t_trunc,
            bin_width=bin_width, num_bins=num_bins)
        cdf = np.cumsum(pmf)
        dt = bin_edges[1] - bin_edges[0]
        t_list = bin_edges[:-1] + (dt-1)/2.
        axs[0][0].plot(t_list, cdf, ".")
        axs[0][1].plot(t_list, pmf)

    w_samples_all = samples_data[1]
    for i in range(len(t_samples_all)):
        w_func = compute_werner(
            t_samples_all[i], w_samples_all[i], bin_edges, t_trunc)
        axs[1][0].plot(t_list, w_func, '.')

    plt.tight_layout()
