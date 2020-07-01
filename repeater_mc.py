import logging
import copy
import time
import os
from functools import partial
from multiprocessing import Pool
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
    t_coh = parameters["t_coh"]
    sample_size = parameters["sample_size"]
    if "cut_type" not in parameters:
        cut_type = "memory_time"
    else:
        cut_type = parameters["cut_type"]
    if "tau" in parameters:  # backward compatibility
        parameters["mt_cut"] = parameters.pop("tau")
    if "cutoff_dict" in parameters.keys():
        cutoff_dict = parameters["cutoff_dict"]
        mt_cut = cutoff_dict.get("memory_time", np.iinfo(np.int).max)
        w_cut = cutoff_dict.get("fidelity", 1.e-8)
        rt_cut = cutoff_dict.get("run_time", np.iinfo(np.int).max)
    else:
        mt_cut = parameters.get("mt_cut", np.iinfo(np.int).max)
        w_cut = parameters.get("w_cut", 1.e-8)
        rt_cut = parameters.get("rt_cut", np.iinfo(np.int).max)
    if not isinstance(mt_cut, Iterable):
        mt_cut = (mt_cut,) * len(protocol)
    else:
        mt_cut = tuple(mt_cut)
    if not isinstance(w_cut, Iterable):
        w_cut = (w_cut,) * len(protocol)
    else:
        w_cut = tuple(w_cut)
    if not isinstance(rt_cut, Iterable):
        rt_cut = (rt_cut,) * len(protocol)
    else:
        rt_cut = tuple(rt_cut)
    network_parameters = (protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type)

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
    w_func = compute_werner(t_samples_list, w_samples_list, bin_edges, t_trunc)
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
    protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = network_parameters

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
    protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = \
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
def cut_off_swap(step, network_parameters):
    """
    Simulate the time out protocol that discards both link when waiting time
    difference is larger than the cut-off mt_cut. Both qubits are discarded when
    it fails.
    """
    protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = network_parameters
    mt_cut = mt_cut[step]
    w_cut = w_cut[step]
    rt_cut = rt_cut[step]

    tA, tB = 0, 0
    wA, wB = 1.0, 1.0
    t_tot = 0.
    cut_off_pass = False
    if cut_type == "memory_time":
        while not cut_off_pass:
            tA, wA = sample_protocol(step-1, network_parameters)
            tB, wB = sample_protocol(step-1, network_parameters)
            t_tot += mt_cut + min(tA, tB)
            cut_off_pass = np.abs(tA - tB) <= mt_cut
        t_tot -= mt_cut
        t_tot += abs(tA - tB)
    elif cut_type == "fidelity":
        while not cut_off_pass:
            tA, wA = sample_protocol(step-1, network_parameters)
            tB, wB = sample_protocol(step-1, network_parameters)
            if tA < tB:
                if wA < w_cut:
                    waiting_time = tA
                else:
                    waiting_time = min(tB, tA + np.int(np.floor(t_coh * np.log(wA/w_cut))))
            elif tA > tB:
                if wB < w_cut:
                    waiting_time = tB
                else:
                    memory_time = np.int(np.floor(t_coh * np.log(wB/w_cut)))
                    waiting_time = min(tA, tB + memory_time)
            else:
                waiting_time = tA

            if tA <= tB and wA * np.exp(-np.abs(tB - tA)/t_coh) >= w_cut and wB>=w_cut:
                cut_off_pass = True
            elif tA > tB and wB * np.exp(-np.abs(tA - tB)/t_coh) >= w_cut and wA>=w_cut:
                cut_off_pass = True
            else:
                cut_off_pass = False
            t_tot += waiting_time
    elif cut_type == "run_time":
        while not cut_off_pass:
            tA, wA = sample_protocol(step-1, network_parameters)
            tB, wB = sample_protocol(step-1, network_parameters)
            if tA > rt_cut or tB > rt_cut:
                cut_off_pass = False
                t_tot += rt_cut
            else:
                cut_off_pass = True
                t_tot += max(tA, tB)
    if tA < tB:
        wA *= np.exp(-(tB-tA)/t_coh)
    else:
        wB *= np.exp(-(tA-tB)/t_coh)
    return (t_tot, tA, tB, wA, wB)


# @nb.jit(nopython=True)
# def cut_off_dist(step, network_parameters):
#     """
#     This should be keep identical to cut_off_swap.
#     This duplication exists
#     because numba is not stable with complicated recursive behavior.
#     Both qubits are discarded when it fails.
#     """
#     protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = network_parameters
#     mt_cut = mt_cut[step]
#     w_cut = w_cut[step]

#     tA, tB = 0, 0
#     wA, wB = 1.0, 1.0
#     t_tot = 0.
#     cut_off_pass = False
#     if cut_type == "memory_time":
#         while not cut_off_pass:
#             tA, wA = sample_protocol(step-1, network_parameters)
#             tB, wB = sample_protocol(step-1, network_parameters)
#             t_tot += mt_cut + min(tA, tB)
#             cut_off_pass = np.abs(tA - tB) <= mt_cut
#         t_tot -= mt_cut
#         t_tot += abs(tA - tB)
#     elif cut_type == "fidelity":
#         while not cut_off_pass:
#             tA, wA = sample_protocol(step-1, network_parameters)
#             tB, wB = sample_protocol(step-1, network_parameters)
#             if tA < tB:
#                 if wA < w_cut:
#                     waiting_time = tA
#                 else:
#                     waiting_time = min(tB, tA + np.int(np.floor(t_coh * np.log(wA/w_cut))))
#             elif tA > tB:
#                 if wB < w_cut:
#                     waiting_time = tB
#                 else:
#                     memory_time = np.int(np.ceil(t_coh * np.log(wB/w_cut)))
#                     waiting_time = min(tA, tB + memory_time)
#             else:
#                 waiting_time = tA

#             if tA <= tB and wA * np.exp(-np.abs(tB - tA)/t_coh) >= w_cut and wB>=w_cut:
#                 cut_off_pass = True
#             elif tA > tB and wB * np.exp(-np.abs(tA - tB)/t_coh) >= w_cut and wA>=w_cut:
#                 cut_off_pass = True
#             else:
#                 cut_off_pass = False
#             t_tot += waiting_time
#     # elif cut_type == "run_time":
#     #     while not cut_off_pass:
#     #         tA, wA = sample_protocol(step-1, network_parameters)
#     #         tB, wB = sample_protocol(step-1, network_parameters)
#     #         if tA >
#     if tA < tB:
#         wA *= np.exp(-(tB-tA)/t_coh)
#     else:
#         wB *= np.exp(-(tA-tB)/t_coh)
#     return (t_tot, tA, tB, wA, wB)


@nb.jit(nopython=True)
def sample_swap(step, network_parameters):
    """
    Simulate the entanglement swap recursively with cut-off time.
    """
    protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = network_parameters

    t_tot, tA, tB, wA, wB = cut_off_swap(step, network_parameters)
    w = wA * wB 

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
    protocol, p_gen, p_swap, mt_cut, w_cut, rt_cut, w0, t_coh, cut_type = network_parameters

    t_tot, tA, tB, wA, wB = cut_off_swap(step, network_parameters)

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
        t_trunc = int(np.ceil(np.max(np.concatenate(t_samples_all)) / 2))
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

if __name__ == "__main__":
    parameters = {
        "protocol": (1, ),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.85,
        "tau": (17, 32, 55),
        "w_cut": 0.8,
        "t_coh": 400,
        "t_trunc": 1000,
        "cut_type": "fidelity",
        "sample_size": 100,
        }
    # parameters = {
    #     "protocol": (0, 0, 0),
    #     "p_gen": 0.5,
    #     "p_swap": 0.8,
    #     "tau": 5,
    #     "sample_size": 200000,
    #     "w0": 1.,
    #     "t_coh": 30,
    #     "t_trunc": 100
    #     }
    fig, axs = plt.subplots(2, 2)

    # simulation part
    t_sample_list = []
    w_sample_list = []

    start = time.time()
    print("Sample parameters:")
    print(parameters)
    t_samples_level, w_samples_level = repeater_mc(parameters)
    t_sample_list.append(t_samples_level)
    w_sample_list.append(w_samples_level)
    end = time.time()
    print("MC Simulation elapse time\n", end-start)
    print()

    plot_mc_simulation(
        [t_sample_list, w_sample_list], axs,
        parameters=parameters, bin_width=1)