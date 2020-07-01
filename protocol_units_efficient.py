import logging

import numpy as np
import numba as nb
from logging_utilities import mytimeit


@nb.jit(nopython=True, error_model="python")
def get_one_array(pmf1, pmf2, w_func1, w_func2, t_coh):
    return pmf1, pmf1, pmf2, pmf2


@nb.jit(nopython=True, error_model="python")
def get_w1w2_array(pmf1, pmf2, w_func1, w_func2, t_coh):
    # t1 < t2
    # Pr(t1) Pr(t2) w1 w2 exp(- t2 + t1) = [Pr(t1) w1 exp(t1)] * [Pr(t2) w2 exp(-t2)]
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh) # exp(-t)
    func1_later = pmf1 * w_func1 * decay_factors
    func1_early = pmf1 * w_func1 / decay_factors
    func2_later = pmf2 * w_func2 * decay_factors
    func2_early = pmf2 * w_func2 / decay_factors
    return func1_early, func1_later, func2_early, func2_later


@nb.jit(nopython=True, error_model="python")
def get_w1_array(pmf1, pmf2, w_func1, w_func2, t_coh):
    size = len(pmf1)
    # t1 < t2
    # Pr(t1) Pr(t2) w1 exp(- t2 + t1) = [Pr(t1) w1 exp(t1)] * [Pr(t2) exp(-t2)]
    # t1 > t2
    # Pr(t1) Pr(t2) w1 = [Pr(t1) w1] * Pr(t2)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    func1_later = pmf1 * w_func1
    func2_early = pmf2
    func1_early = pmf1 * w_func1 / decay_factors
    func2_later = pmf2 * decay_factors
    return func1_early, func1_later, func2_early, func2_later


@nb.jit(nopython=True, error_model="python")
def get_w2_array(pmf1, pmf2, w_func1, w_func2, t_coh):
    size = len(pmf1)
    decay_factors = np.exp(- np.arange(size) / t_coh)
    func1_later = pmf1 * decay_factors
    func2_early = pmf2 * w_func2 / decay_factors
    func1_early = pmf1
    func2_later = pmf2 * w_func2
    return func1_early, func1_later, func2_early, func2_later


def join_links_efficient(
        pmf1, pmf2, w_func1, w_func2,
        mt_cut=np.iinfo(np.int32).max, w_cut=0.0, rt_cut=np.iinfo(np.int32).max, ycut=True,
        cut_type=None, evaluate_func=None, t_coh=np.inf):
    if evaluate_func == "1":
        evaluate_coeff_list = (1,)
        final_coeff = 1.
        evaluate_func_list = (get_one_array,)
    elif evaluate_func == "w1w2":
        evaluate_coeff_list = (1,)
        final_coeff = 1.
        evaluate_func_list = (get_w1w2_array,)
    elif evaluate_func == "0.5+0.5w1w2":
        evaluate_coeff_list = (0.5, 0.5)
        final_coeff = 1.
        evaluate_func_list = (get_one_array, get_w1w2_array)
    elif evaluate_func == "0.5-0.5w1w2":
        evaluate_coeff_list = (0.5, -0.5)
        final_coeff = 1.
        evaluate_func_list = (get_one_array, get_w1w2_array)
    elif evaluate_func == "w1+w2+4w1w2":
        evaluate_coeff_list = (1., 1., 4.)
        final_coeff = 1./6.
        evaluate_func_list = (get_w1_array, get_w2_array, get_w1w2_array)
    elif isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    
    size = len(pmf1)
    if size/t_coh > 300:
        logging.warn("Overflow in the exp function!")
    final_result = np.zeros(size, dtype=np.float64)
    for evaluate_coeff, evaluate_func in zip(evaluate_coeff_list, evaluate_func_list):
        # separate the positive and negative part for numerical stability
        result = join_links_efficient_helper(
            pmf1, pmf2, w_func1, w_func2, t_coh, mt_cut, ycut, evaluate_func)
        final_result += evaluate_coeff * result
    return final_coeff * final_result


@nb.jit(nopython=True, error_model="python")
def join_links_efficient_helper(pmf1, pmf2, w_func1, w_func2, t_coh, mt_cut, ycut, evaluate_func):
    size = len(pmf1)
    result = np.zeros(size, dtype=np.float64)
    minus_result = np.zeros(size, dtype=np.float64)
    func1_early, func1_later, func2_early, func2_later = evaluate_func(pmf1, pmf2, w_func1, w_func2, t_coh)
    if ycut:
        # waiting time is max(t1, t2), fix the later link as t
        cum_func1_early = np.cumsum(func1_early)
        cum_func2_early = np.cumsum(func2_early)
        for t in range(1, size):
            cut = max(0, t - mt_cut - 1)
            # link 2 wait
            result[t] += func1_later[t] * cum_func2_early[t]
            minus_result[t] += func1_later[t] * cum_func2_early[cut]
            # link 1 wait
            result[t] += func2_later[t] * cum_func1_early[t-1] 
            minus_result[t] += func2_later[t] * cum_func1_early[cut]
    else:
        # waiting time is min(t1, t2), fix the early link as t
        cum_func1_later = np.cumsum(func1_later)
        cum_func2_later = np.cumsum(func2_later)
        for t in range(1, size - mt_cut):
            cut = t + mt_cut
            # link 1 wait
            result[t] += func2_early[t] * cum_func1_later[-1]
            minus_result[t] += func2_later[t] * cum_func1_later[cut]
            # link 2 wait
            result[t] += func1_early[t] * cum_func2_later[-1]
            minus_result[t] += func1_later[t] * cum_func2_later[cut]
    result -= minus_result
    return result
