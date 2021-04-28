import time
from copy import deepcopy
import multiprocessing as mp
from collections.abc import Iterable
from itertools import product
from functools import partial
import logging

import numpy as np
from scipy.optimize import differential_evolution

from utility_functions import create_cutoff_dict, secret_key_rate
from logging_utilities import (
    log_init, log_params, log_finish, mytimeit, create_iter_kwargs)
from repeater_algorithm import repeater_sim, compute_unit, plot_algorithm, RepeaterChainSimulation
from repeater_mc import repeater_mc, plot_mc_simulation


__all__ = ["CutoffOptimizer",
    "optimization_tau_wrapper", "parallel_tau_warpper",
    "uniform_tau_pretrain",
    "full_tau_pretrain_high_tau", "full_tau_pretrain"]



def optimization_tau_wrapper(
        cutoffs, func, parameters, merit=None,
        ref_pmf_matrix=None, tracker_data=None,
        **kwargs):
    """
    Wrapper for repeater_sim or repeater_mc. It uses the cut-off
    as explicitly parameter and mutes the warning message
    of the given function.
    It is designed to be usd in the optimizer for cut-off time.
    If an error occurs, error
    message will be logged and `pmf` and `w_func` will be two zero array.

    Parameters
    ----------
    cutoffs: array-like 1d
        The memory cut-off time. 
        If no `ref_pmf_matrix` is given, each element should be
        a integer number, otherwise float number.
    func: python function
        A python function that takes parameters and return
        the `pmf` and `w_func`.
        The function can be e.g. `repeater_sim` and `repeater_mc`.
    parameters: dict
        Dictionary for the network parameters. If present,
        the value of the key `cutoffs` will be overwritten.
    ref_pmf_matrix: array-like 2d, optional
        A function that generate the reference distribution of cutoffs. It is used
        to rescale the search space of cut-off time, which is an unbounded
        integer space. If given, `cutoffs` should be an array of float number, and
        the integer cutoffs will be given by ``np.searchsorted(ref_pmf, cutoffs)``.
        It is required that ``len(ref_pmf_matrix)==len(cutoffs)``
    merit: function, optional
        The merit function of the optimization.
        It should take the pmf and w_func
        as input and return a float number. E.g. `secrete_key_rate`
    **kwargs:
        additional keyword arguments for repeater_sim and repeater_mc.

    Returns
    -------
    If `merit` is None:
    pmf: array-like 1-D
        The waiting time distribution of the distillation.
    w_func: array-like 1-D
        The Werner parameter as function of T of the distillation.

    If `merit` is given:
    negative_merit: float
        ``- merit(pmf, w_func)``
    """
    parameters = deepcopy(parameters)
    if "cut_type" in parameters:
        cut_type = parameters["cut_type"]
    else:
        cut_type = "memory_time"

    if isinstance(cutoffs, dict):
        cutoff_dict = cutoffs
    else:
        if isinstance(cutoffs, Iterable):
            cutoffs = np.asarray(cutoffs)
        elif np.isscalar(cutoffs):
            cutoffs = np.asarray([cutoffs])
        cutoff_dict = create_cutoff_dict(cutoffs, cut_type, parameters, ref_pmf_matrix)

    parameters["cutoff_dict"]= cutoff_dict

    # suppress the truncation time warning, we check it seperately.
    current_log_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    try:
        pmf, w_func = func(parameters=parameters, **kwargs)
    except Exception as err:
        logging.exception(
            "Running the simulation fails with the following parameter:\n" + 
            str(parameters))
        raise err
    logging.getLogger().setLevel(current_log_level)

    if merit is not None:
        merit_result = 0. - merit(pmf, w_func)
    if merit is not None:
        return merit_result
    else: 
        return pmf, w_func


def parallel_tau_warpper(tau_list, parameters, func=None, t_trunc=None, workers=1):
    """
    An additional wrapper that enables multi-processing.

    Parameters
    ----------
    tau_list: int or array-like
        A list of tau, see paramters of `optimization_tau_wrapper`.
    parameters: dict
        Dictionary for the network parameters. The value of key `tau` will be
        overwritten by parameter `tau`.
    t_trunc: int
        Truncation time of the simulation

    Returns
    -------
    pmf_list: array-like 2-D
        The waiting time distribution of the distillation for different tau.
    w_func_list: array-like 2-D
        The Werner parameter as function of T of the distillation
        for different tau.
    """
    if func is None:
        func = repeater_sim
    pmf_list = []
    w_func_list = []
    if workers == 1:
        result = map(
            partial(
                optimization_tau_wrapper,
                func=func,
                parameters=parameters
                ),
            tau_list)
    else:
        pool = mp.Pool(workers)
        result = pool.map(
            partial(
                optimization_tau_wrapper,
                func=func,
                parameters=parameters
                ),
            tau_list)
        pool.close()
        pool.join()
    for _, (pmf, w_func) in enumerate(result):
        pmf_list.append(pmf)
        w_func_list.append(w_func)

    return pmf_list, w_func_list


def call_back(xk, convergence):
    print("current cut-off location", xk)
    print("convergence:", convergence)
    return None


class CutoffOptimizer():
    """
    Optimizer for the cut-offs. It takes the simulation parameters and
    find the optimal cut-off using a heuristic differential evolution
    algorithm.

    Parameters
    ----------
    opt_kind: str, optional
        `nonuniform_de` for non-uniform cut-off or
        `uniform_de` for uniform cut-off
    adaptive: bool, optional
        If the found cut-off is not a local optimal because of
        the discrete search space, improve the optimization parameters
        and restart the algorithm
    wokers:, optional
        Number of processes used for parallel computing
        (`multiprocessing.Pool`) for differential evolution.
    pretrain: python function, optional
        The pretraining function for the reference waiting time distribution.
    sample_distance: int, optional
        Distance of the sampled cut-off used in checking if the result is optimal.
        After the differential evolution algorithm terminates,
        we perform a local optimality check.
        We compare `cutoff` with ``cutoff-sample_distance``
        and ``cutoff+sample_distance``.
        If the difference is smaller than 0.001%,
        the check succeed.
        If you want to find the very best cut-off, set this to 1.
        Default is `max(1, int(t_trunc/10000))`.
    **de_wargs:
        Additional key word arguments for differential evolution.

    See Also
    --------
    [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    """
    def __init__(
            self, opt_kind="nonuniform_de", adaptive=False, workers=None,
            pretrain=None, sample_distance=None, simulator=None, **de_kwargs):
        self.opt_kind = opt_kind
        if pretrain is None:
            if self.opt_kind == "nonuniform_de":
                self.pretrain = full_tau_pretrain
            if self.opt_kind == "uniform_de":
                self.pretrain = uniform_tau_pretrain
        else:
            self.pretrain = pretrain
        self.adaptive = adaptive
        self.de_kwargs = de_kwargs
        if workers is None:
            self.workers = mp.cpu_count() - 2
            if self.workers <= 0:
                self.workers = 1
        else:
            self.workers = workers
        self.sample_distance = sample_distance
        self.simulator = simulator

    def run(self, parameters):
        """
        Parameters
        ----------
        tau_dims: int or iterable, optional
            Default is different tau for each level of swap and distillation
            according to the given `parameter`.
        """
        logging.info("-------------------------------------------")
        logging.info("Optimization of the cut-off time\n")
        log_params(parameters)
        parameters = deepcopy(parameters)
        # remove cutoff related keywords for safety
        parameters.pop("mt_cut", None)
        parameters.pop("w_cut", None)
        parameters.pop("rt_cut", None)
        parameters.pop("cutoff", None)
        parameters.pop("cutoff_dict", None)
        if "cut_type" in parameters:
            self.cut_type = parameters["cut_type"]
        else:
            parameters["cut_type"] = "memory_time"
            self.cut_type = "memory_time"
        if self.opt_kind == "uniform_de":
            if self.cut_type != "memory_time":
                raise UserWarning("Only memory time has good performance with uniform cutoff.")
            tau_dims = 1
        elif self.opt_kind == "nonuniform_de":
            tau_dims = len(parameters["protocol"])
        else:
            raise ValueError("Unknown optimization method")

        # pretraining, using a known distribution instead of uniform sampling
        # to speed up the convergence.
        # Similar to sklearn.preprocessing.quantile_transform.
        if self.cut_type in ("memory_time", "run_time"):
            logging.info("Pretraining begins...")
            ref_pmf_matrix = self.pretrain(parameters, tau_dims=tau_dims)
            logging.info("Pretraining finishes, reference pmf obtained.")
        else:
            ref_pmf_matrix = None

        # DE optimization
        de_config = {
            # Default config, the following parameters can be changed by
            # de_kwargs
            "bounds": [(0., 1.)] * tau_dims,
            "updating": "deferred",
            "disp": True,
            "workers": self.workers,
            "strategy": "best1exp",
            "callback": call_back,
            "tol": 0.01,
            "popsize": 10
        }
        if logging.getLogger().level == logging.DEBUG:
            de_config["workers"] = 1
        de_config.update(self.de_kwargs)  # de_kwargs has privilege

        count = 0  # Number of repetitions in total
        if self.simulator is None:
            self.simulator = RepeaterChainSimulation()
        while True:
            target_function = partial(
                optimization_tau_wrapper,
                func=self.simulator.nested_protocol,
                parameters=parameters,
                merit=secret_key_rate,
                ref_pmf_matrix=ref_pmf_matrix,
                )
            result = differential_evolution(
                target_function,
                **de_config
            )

            # Processing result
            best_raw_cutoffs = result.x
            best_cutoff_dict = create_cutoff_dict(best_raw_cutoffs, self.cut_type, parameters, ref_pmf_matrix)
            best_pmf, best_w_func = optimization_tau_wrapper(
                cutoffs=best_cutoff_dict,
                func=self.simulator.nested_protocol,
                parameters=parameters
                )
            best_key_rate = secret_key_rate(best_pmf, best_w_func)

            # Check if succeeds
            nonzero_rate, nonzero_rate_warning_msg = self.nonzero_rate_check(
                best_key_rate)
            if "memory_time" in best_cutoff_dict.keys():
                if self.sample_distance is None:
                    self.sample_distance = max(1, int(parameters["t_trunc"] / 10000))
                local_max_check, local_max_check_warning_msg = \
                    self.check_local_max(
                        parameters, best_cutoff_dict, self.sample_distance)
            else:
                local_max_check, local_max_check_warning_msg = True, ""
            coverage_check, coverage_check_warning_msg = self.check_coverage(
                best_pmf)
            # We don't include coverage check because the secret key rate
            # depends on it and, hence, we cannot increase it.
            terminate = nonzero_rate and local_max_check

            # Check if terminates
            count += 1
            if count >= 10:
                logging.warning(
                    "Maximal number of attempts arrived. "
                    "Optimization fails.")
                break
            if not self.adaptive or terminate:
                break

            # Optimization fails, we make some adaption and restart
            logging.info(
                nonzero_rate_warning_msg +
                # coverage_check_warning_msg +
                local_max_check_warning_msg)
            logging.info(
                f"The current cut-off found: {best_cutoff_dict}\n"
                f"The current cut-off is located at: {best_raw_cutoffs}\n"
                f"The current key rate is {best_key_rate}")
            logging.info(
                "The following change has been made to the parameters:")
            # if not coverage_check:
            #     parameters["t_trunc"] = self.increase_trunc(
            #         parameters["t_trunc"])
            #     ref_pmf_matrix = self.pretrain(parameters, tau_dims)
            if not nonzero_rate:
                ref_pmf_matrix = self.reduce_lower_limit(
                    ref_pmf_matrix)
            elif not local_max_check:
                de_config["popsize"] = self.increase_popsize(
                    de_config["popsize"])
                ref_pmf_matrix = self.restrict_search_region(
                    ref_pmf_matrix, best_raw_cutoffs)
                de_config["tol"] = de_config["tol"]/5.
            logging.info(
                "Optimization fails to find the best cut-off. Restarting.\n")

        warning_msg = (
            nonzero_rate_warning_msg + 
            coverage_check_warning_msg + 
            local_max_check_warning_msg)
        if warning_msg != "":
            logging.warning(str(parameters) + "\n" + warning_msg)
        logging.info(
            f"The best cut-off found: {best_cutoff_dict}\n"
            f"The best cut-off is located at: {best_raw_cutoffs}\n"
            f"The best key rate is {best_key_rate}\n")

        logging.info("-------------------------------------------")

        return best_cutoff_dict

    def nonzero_rate_check(self, best_key_rate):
        """
        Check if the merit is zero.
        """
        if best_key_rate == 0.0:
            nonzero_rate_warning_msg = (
                "The best key rate after the optimization is still 0. "
                "This may indicate that the cut-off time required is "
                "very small, please adjust the range of the reference "
                "distribution and give more weight to small cut-off.\n")
            nonzero_rate = False
        else:
            nonzero_rate_warning_msg = ""
            nonzero_rate = True
        return nonzero_rate, nonzero_rate_warning_msg

    def check_coverage(self, best_pmf):
        """
        Check if the merit has cover enough probability distribution of
        the waiting time.
        """
        if np.sum(best_pmf) < 0.99:
            coverage_check = False
            coverage_check_warning_msg = (
                "The probability coverage is only {:.2}%, please check "
                "the validity of the result by increasing t_trunc.\n".format(
                    100 * np.sum(best_pmf)))
        else:
            coverage_check = True
            coverage_check_warning_msg = ""
        return coverage_check, coverage_check_warning_msg

    def check_local_max(self, parameters, cutoff_dict, sample_distance=1):
        """
        Check if the curret found cut-off the locally optimal.
        It check all the direct neighbours, if the difference between
        maximal merit and the current merit is smaller than the tolerance.
        The check passes.
        Only works for discrete cut-off time.
        """
        parameters = deepcopy(parameters)
        time_cutoff = np.concatenate([
            cutoff_dict.get("memory_time", []),
            cutoff_dict.get("run_time", [])
            ])
        if parameters.get("cut_type", "memory_time") == "run_time":
            for i in range(1, len(time_cutoff)):
                time_cutoff[i] = time_cutoff[i] - np.sum(time_cutoff[:i])
        cutoff_with_neighbor = []
        if self.opt_kind == "uniform_de":
            time_cutoff = time_cutoff[0:1]
        for t in time_cutoff:
            t_trunc = parameters["t_trunc"]
            t_max = min(t + sample_distance, t_trunc)
            t_min = max(t - sample_distance, 0)
            cutoff_with_neighbor.append([t_min, t, t_max])

        # iteration for all time_cutoff combination in cutoff_with_neighbor
        tau_list = [time_cutoff] + list(product(*cutoff_with_neighbor))
        pmf_list, w_func_list = parallel_tau_warpper(
            tau_list, parameters, func=self.simulator.nested_protocol,
            workers=self.workers)
        key_rate_list = [
            secret_key_rate(pmf, w_func)
            for pmf, w_func in zip(pmf_list, w_func_list)]
        if np.argmax(key_rate_list) < 1.0e-10:  # key rates are all 0
            return True, ""
        better_iter = np.argmax(key_rate_list)
        better_cutoff = tau_list[better_iter]
        better_rate = key_rate_list[better_iter]
        increase = (better_rate - key_rate_list[0])/key_rate_list[0]
        better_cutoff = create_cutoff_dict(better_cutoff, parameters["cut_type"], parameters)
        if increase > 1.e-5:
            warning_msg = (
                "Local optimal check fails. "
                "The cut-off found is not optimal. "
                "Neighboring cut-off {0} is better with an increase "
                "in the secrete key rate of {1:.3f}%.\n".format(
                    better_cutoff, increase * 100
                    )
                )
            return False, warning_msg
        else:
            return True, ""

    def increase_trunc(self, t_trunc):
        """
        Increase the truncation time. Used when the coverage is too low.
        """
        t_trunc = t_trunc * 4 // 3
        logging.info(
            f"t_trunc has been increased to {t_trunc} for more coverage. "
            "Please watch out for the running time.")
        return t_trunc

    def reduce_lower_limit(self, ref_pmf_matrix):
        """
        Effective only if use time cut-off with a reference probability.
        Shift the reference probability distribution to the left so that
        small cut-off time has larger weight.
        Used when the cut-off is very small.
        """
        if ref_pmf_matrix is None:
            return ref_pmf_matrix
        new_ref_pmf_matrix = ref_pmf_matrix[:, 0::2]
        if ref_pmf_matrix.shape[1] % 2 == 0:
            new_ref_pmf_matrix += ref_pmf_matrix[:, 1::2]
        else:
            new_ref_pmf_matrix[:, :-1] += \
                new_ref_pmf_matrix[:, :-1] + ref_pmf_matrix[:, 1::2]
        logging.info(
            "The cut-off seems to be very small, "
            "the reference pmf has been changed to rescale the search space.")
        return new_ref_pmf_matrix

    def restrict_search_region(self, ref_pmf_matrix, best_tau_loc):
        """
        Restrict the search region.
        Used when the local maximum check fails.
        """
        for i, (t, ref_pmf) in enumerate(zip(best_tau_loc, ref_pmf_matrix)):
            ref_cmf = np.cumsum(ref_pmf)
            t_prob = best_tau_loc[i]
            prob_max = min(1.0, t_prob + 0.3)
            t_max = np.searchsorted(ref_cmf, prob_max) + 1
            prob_min = max(0.0, t_prob - 0.3)
            t_min = np.searchsorted(ref_cmf, prob_min)
            if (t_max - t_min) >= 10:
                ref_pmf[0: t_min] = 0.
                ref_pmf[t_max:] = 0.
                ref_pmf_matrix[i] = 0.991 / np.sum(ref_pmf) * ref_pmf
                logging.info(
                    f"Search region for cutoff[{i}] is restricted to "
                    f"({t_min},{t_max}).")
        return ref_pmf_matrix

    def increase_popsize(self, popsize):
        """
        Increase the number of populations.
        Used when the local maximum check fails
        """
        popsize = popsize * 3 // 2
        logging.info("'popsize' is increased to {}.".format(popsize))
        return popsize


def pretrain_wrapper(func):
    """
    Wrapper for the pretraining functions. Disable the warnings.
    """
    def inner(parameters, tau_dims, **kwargs):
        parameters = deepcopy(parameters)
        current_log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        ref_pmf_matrix = func(parameters, tau_dims=tau_dims, **kwargs)
        msg = False
        while np.sum(ref_pmf_matrix[-1]) < 0.99:
            msg = True
            parameters["t_trunc"] = parameters["t_trunc"] * 4 // 3
            ref_pmf_matrix = func(parameters, tau_dims=tau_dims, **kwargs)
        logging.getLogger().setLevel(current_log_level)
        if msg:
            logging.info(
                "Not enough probability is covered for the pretraining, "
                "the truncation time is increased to {}.".format(
                    parameters["t_trunc"]))
        return ref_pmf_matrix
    return inner


@pretrain_wrapper
def uniform_tau_pretrain(parameters, tau_dims):
    """
    Return the probability distribution of the highest level.
    """
    ref_pmf, _ = optimization_tau_wrapper(
        [np.iinfo(np.int32).max] * tau_dims,
        func=repeater_sim, parameters=parameters)
    return np.array([ref_pmf] * tau_dims)


@pretrain_wrapper
def full_tau_pretrain_high_tau(parameters, tau_dims):
    """
    Return the probability distribution without cut-off of level 1 to n.
    """
    parameters["tau"] = (np.iinfo(np.int32).max,) * len(parameters["protocol"])
    full_result = repeater_sim(parameters, all_level=True)
    ref_pmf_matrix = np.array([result_pair[0] for result_pair in full_result])
    return ref_pmf_matrix[1:]


@pretrain_wrapper
def full_tau_pretrain(parameters, tau_dims):
    """
    Return the probability distribution without cut-off of level 0 to n-1.
    """
    parameters["tau"] = (np.iinfo(np.int32).max,) * len(parameters["protocol"])
    full_result = repeater_sim(parameters, all_level=True)
    ref_pmf_matrix = np.array([result_pair[0] for result_pair in full_result])
    return ref_pmf_matrix


@pretrain_wrapper
def guess_tau_pretrain(parameters, tau_dims, geuss_tau):
    """
    Return the probability distribution with a given cut-off of level 0 to n-1.
    """
    parameters["tau"] = geuss_tau
    full_result = repeater_sim(parameters, all_level=True)
    ref_pmf_matrix = np.array([result_pair[0] for result_pair in full_result])
    return ref_pmf_matrix[:-1]


if __name__ == "__main__":
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.002,
        "p_swap": 0.25,
        "w0": 0.97,
        "t_coh": 35000,
        "t_trunc": 2000000,
        "cut_type": "memory_time",
        "sample_distance": 50,
        "tol": 0.0001,
        }

    ID = log_init("optimize", level=logging.INFO)
    simulator = RepeaterChainSimulation()
    simulator.use_gpu = True
    optimizer = CutoffOptimizer(simulator=simulator, workers=8, adaptive=True, opt_kind="nonuniform_de")
    optimal_cutoff = optimizer.run(parameters)