"""
This file contains the optimizer class for cut-off, CutoffOptimizer
"""
import time
from copy import deepcopy
import multiprocessing as mp
from collections.abc import Iterable
from itertools import product
from functools import partial
import logging

import numpy as np
from scipy.optimize import differential_evolution

from utility_functions import secret_key_rate
from logging_utilities import (
    log_init, log_params, log_finish, mytimeit, create_iter_kwargs)
from repeater_algorithm import repeater_sim, compute_unit, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation


__all__ = ["CutoffOptimizer",
    "optimization_tau_wrapper", "parallel_tau_warpper",
    "uniform_tau_pretrain",
    "full_tau_pretrain_high_tau", "full_tau_pretrain_low_tau"]


def optimization_tau_wrapper(
        tau, func, parameters, merit=None,
        ref_pmf_matrix=None, tracker_data=None, **kwargs):
    """
    Wrapper for repeater_sim or repeater_mc. It uses the cut-off time
    tau as explicitly parameter and mutes the warning message
    of the given function.
    It is designed to be usd in the optimizer for cut-off time.
    If an error occurs, error
    message will be logged and `pmf` and `w_func` will be two zero array.

    Parameters
    ----------
    tau: array-like 1d
        The memory cut-off time. 
        If no `ref_pmf_matrix` is given, each element should be
        a integer number, otherwise float number.
    func: repeater_mc or repeater_sim
        The Monte Carlo algorithm or the deterministic algorithm.
    parameters: dict
        Dictionary for the network parameters. If present,
        the value of the key `tau` will be overwritten.
    ref_pmf_matrix: array-like 2d, optional
        A function that generate the reference distribution of tau. It is used
        to rescale the search space of cut-off time, which is an unbounded
        integer space. If given, `tau` should be an array of float number, and
        the integer tau will be given by ``np.searchsorted(ref_pmf, tau)``.
        It is required that ``len(ref_pmf_matrix)==len(tau)``
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
        ``0. - merit(pmf, w_func)``
    """
    parameters = deepcopy(parameters)

    if isinstance(tau, Iterable):
        tau = np.array(tau)
    if np.isscalar(tau):
        tau = np.array([tau])

    if ref_pmf_matrix is not None:
        if all(tau < 0.) or all(tau > 1.):
            raise ValueError(
                "A reference pmf is given, but tau is not a probability")
        if len(tau) != len(ref_pmf_matrix):
            raise ValueError(
                "The reference probability matrix must have "
                "the same length as the input tau. However\n "
                "len(tau)={}\n len(ref_pmf_matrix)={}\n".format(
                    len(tau), len(ref_pmf_matrix)))
        tau_pos = tau
        tau = np.empty(tau_pos.shape, dtype=np.int)
        for i in range(len(tau)):
            tau[i] = np.searchsorted(np.cumsum(ref_pmf_matrix[i]), tau_pos[i])
        logging.debug("The input tau postions are {}:".format(tau_pos))

    logging.debug("The input tau is {}".format(tau))

    # level by level optimization tau must be an integer,
    # but differential evolution generate a np.ndarray of
    # length one automatically.
    tau = tuple(tau)
    if func == compute_unit:
        if len(tau) != 1:
            raise ValueError("For compute_unit tau must have length 1.")
        else:
            tau = tau[0]
    if len(tau) == 1 and len(parameters["protocol"]) != 1:
        tau = tau * len(parameters["protocol"])
    parameters["tau"] = tau

    # if tracker_data is not None and merit is not None:
    #     merit_result = get_merit_record(tracker_data, parameters)
    #     if merit_result is not None:
    #         return merit_result

    # suppress the truncation time warning, we check it seperately.
    # TODO use logging handler to deal with this
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

    coverage = np.sum(pmf)
    if merit is not None:
        if coverage >= 0.95:
            merit_result = 0. - merit(pmf, w_func)
        else:
            merit_result = 0.
    # if tracker_data is not None:
    #     if coverage > 0.99:
    #         add_merit_record(tracker_data, parameters, merit_result)

    if merit is not None:
        return merit_result
    else: 
        return pmf, w_func


def parallel_tau_warpper(tau_list, parameters, t_trunc=None, workers=1):
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
    if t_trunc is None:
        t_trunc = parameters["t_trunc"]
    pmf_list = []
    w_func_list = []
    if workers == 1:
        result = map(
            partial(
                optimization_tau_wrapper,
                func=repeater_sim,
                parameters=parameters
                ),
            tau_list)
    else:
        pool = mp.Pool(workers)
        result = pool.map(
            partial(
                optimization_tau_wrapper,
                func=repeater_sim,
                parameters=parameters
                ),
            tau_list)
        pool.close()
        pool.join()
    for tau_ind, (pmf, w_func) in enumerate(result):
        pmf_list.append(pmf)
        w_func_list.append(w_func)

    return pmf_list, w_func_list


def call_back(xk, convergence):
    # print(xk)
    print("convergence:", convergence)
    return None


class CutoffOptimizer():
    """
    Optimizer for the cut-offs. It takes the simulation parameters and
    find the optimal cut-off using a heuristic differential evolution
    algorithm.

    Parameters
    ----------
    opt_kind: str
        "full_de" for non-uniform cut-off or
        "unform_de" for uniform cut-off
    adaptive: bool
        If the found cut-off is not a local optimal because of
        the discrete search space, improve the optimization parameters
        and restart the algorithm
    wokers:
        Number of processes used for parallel computing
        (`multiprocessing.Pool`) for differential evolution.
    tolerance:
        If the search space is discrete. This is the absolute
        tolerance for the slop.
        The difference in the merit between the result and
        the nearest neighbour must smaller than the tolerance.
        Be careful, this is NOT the tolerance for the maximum.
        To set the tolerance for differential evolution,
        please use `de_kwargs`.
    pretrain: python function
        The pretraining function for the reference waiting time distribution.
    **de_wargs:
        Additional key word argument for differential evolution

    See Also
    --------
    [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    """
    def __init__(
            self, opt_kind="full_de",
            tolerance=0., adaptive=False, workers=None,
            pretrain=None, use_tracker=False, **de_kwargs):
        self.opt_kind = opt_kind
        if pretrain is None:
            if self.opt_kind == "full_de":
                self.pretrain = full_tau_pretrain_low_tau
            if self.opt_kind == "uniform_de":
                self.pretrain = uniform_tau_pretrain
        else:
            self.pretrain = pretrain
        self.use_tracker = use_tracker
        self.tolerance = tolerance
        self.adaptive = adaptive
        self.de_kwargs = de_kwargs
        if workers is None:
            self.workers = mp.cpu_count() - 1
        else:
            self.workers = workers

    @mytimeit
    def run(self, parameters, tau_dims=None):
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
        if tau_dims is None:
            tau_dims = len(parameters["protocol"])
        else:
            tau_dims = tau_dims

        # pretraining
        logging.info("Pretraining begins...")
        ref_pmf_matrix = self.pretrain(parameters, tau_dims=tau_dims)
        logging.info("Pretraining finishes, reference pmf obtained.")

        # DE optimization
        de_config = {
            # Default config, the following parameters can be changed by
            # de_kwargs
            "bounds": [(0.01, 1.)] * tau_dims,
            "updating": "deferred",
            "disp": True,
            "workers": self.workers,
            "strategy": "best1exp",
            "callback": call_back,
            "popsize": 10
        }
        if logging.getLogger().level == logging.DEBUG:
            de_config["workers"] = 1
        de_config.update(self.de_kwargs)  # de_kwargs has privilege

        count = 0  # Number of repetitions in total
        if self.use_tracker:
            tracker_data = {}
        else:
            tracker_data = None
        while True:
            target_function = partial(
                optimization_tau_wrapper,
                func=repeater_sim,
                parameters=parameters,
                merit=secret_key_rate,
                ref_pmf_matrix=ref_pmf_matrix,
                tracker_data=tracker_data)
            result = differential_evolution(
                target_function,
                **de_config
            )

            # Processing result
            best_tau_prob = result.x
            best_tau = np.empty(len(ref_pmf_matrix), dtype=np.int)
            for i in range(len(ref_pmf_matrix)):
                best_tau[i] = np.searchsorted(
                    np.cumsum(ref_pmf_matrix[i]), best_tau_prob[i])
            best_tau = tuple(best_tau)
            if self.opt_kind == "uniform_de":
                assert(len(best_tau) == 1)
                best_tau = best_tau * len(parameters["protocol"])
            best_pmf, best_w_func = optimization_tau_wrapper(
                tau=best_tau,
                func=repeater_sim,
                parameters=parameters
                )
            best_key_rate = secret_key_rate(best_pmf, best_w_func)

            # Check if succeeds
            nonzero_rate, nonzero_rate_warning_msg = self.nonzero_rate_check(
                best_key_rate)
            local_max_check, local_max_check_warning_msg = \
                self.check_local_max(parameters, best_tau, best_tau)
            coverage_check, coverage_check_warning_msg = self.check_coverage(
                best_pmf)
            lower_minimum_reached = any(best_tau_prob < 0.03)
            terminate = nonzero_rate and coverage_check and local_max_check

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
                coverage_check_warning_msg +
                local_max_check_warning_msg)
            logging.info(
                f"The current tau found: {best_tau}\n"
                f"The current cut-off is located at: {best_tau_prob}\n"
                f"The current key rate is {best_key_rate}")
            logging.info(
                "The following change has been made to the parameters:")
            if not coverage_check:
                parameters["t_trunc"] = self.increase_trunc(
                    parameters["t_trunc"])
                ref_pmf_matrix = self.pretrain(parameters, tau_dims)
            elif not nonzero_rate or lower_minimum_reached:
                ref_pmf_matrix = self.reduce_lower_limit(
                    ref_pmf_matrix, best_tau_prob)
            elif not local_max_check:
                de_config["popsize"] = self.increase_popsize(
                    de_config["popsize"])
                ref_pmf_matrix = self.restrict_search_region(
                    ref_pmf_matrix, best_tau)
            logging.info(
                "Optimization fails to find the best cut-off. Restarting.\n")

        warning_msg = (
            nonzero_rate_warning_msg + 
            coverage_check_warning_msg + 
            local_max_check_warning_msg)
        if warning_msg != "":
            logging.warning(str(parameters) + "\n" + warning_msg)
        logging.info(
            f"The best tau found: {best_tau}\n"
            f"The best cut-off is located at: {best_tau_prob}\n"
            f"The best key rate is {best_key_rate}\n")

        logging.info("-------------------------------------------")

        return best_tau

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

    def check_local_max(self, parameters, tau, best_tau_prob):
        """
        Check if the curret found cut-off the locally optimal.
        It check all the direct neighbours, if the difference between
        maximal merit and the current merit is smaller than the tolerance.
        The check passes.
        Only works for discrete cut-off time.
        """
        parameters = deepcopy(parameters)
        tau_with_neighbor = []
        if self.opt_kind == "uniform_de":
            tau = tau[0:1]
        for t in tau:
            if t != 0 and t < parameters["t_trunc"]:
                tau_with_neighbor.append([t-1, t, t + 1])
            elif t == 0:
                tau_with_neighbor.append([t, t+1])
            else:
                tau_with_neighbor.append([t-1, t])

        # iteration for all tau combination in tau_with_neighbor
        tau_list = [tau] + list(product(*tau_with_neighbor))
        pmf_list, w_func_list = parallel_tau_warpper(
            tau_list, parameters, workers=self.workers)
        key_rate_list = [
            secret_key_rate(pmf, w_func)
            for pmf, w_func in zip(pmf_list, w_func_list)]

        if np.argmax(key_rate_list) == 0:
            return True, ""
        better_iter = np.argmax(key_rate_list)
        better_tau = tau_list[better_iter]
        better_rate = key_rate_list[better_iter]
        increase = (better_rate - key_rate_list[0])/key_rate_list[0]
        if increase > self.tolerance:
            warning_msg = (
                "Local optimal check fails. "
                "The best tau found is not optimal. "
                "Neighboring tau = {0} is better with an increase "
                "in the secrete key rate of {1:.2f}%.\n".format(
                    better_tau, increase * 100
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

    def reduce_lower_limit(self, ref_pmf_matrix, best_tau_prob):
        """
        Shift the reference probability distribution to the left so that
        small cut-off time has larger weight.
        Used when the cut-off is very small.
        """
        logging.info(
            "The cut-off seems to be very small, "
            "the reference pmf has been changed to rescale the search space.")
        new_ref_pmf_matrix = ref_pmf_matrix[:, 0::2]
        if ref_pmf_matrix.shape[1] % 2 == 0:
            new_ref_pmf_matrix += ref_pmf_matrix[:, 1::2]
        else:
            new_ref_pmf_matrix[:, :-1] += \
                new_ref_pmf_matrix[:, :-1] + ref_pmf_matrix[:, 1::2]
        return new_ref_pmf_matrix

    def restrict_search_region(self, ref_pmf_matrix, best_tau):
        """
        Restrict the search region.
        Used when the local maximum check fails.
        """
        for i, (t, ref_pmf) in enumerate(zip(best_tau, ref_pmf_matrix)):
            ref_cmf = np.cumsum(ref_pmf)
            t_prob = ref_cmf[best_tau[i]]
            prob_max = min(1.0, t_prob + 0.2)
            t_max = np.searchsorted(ref_cmf, prob_max) + 1
            prob_min = max(0.0, t_prob - 0.2)
            t_min = np.searchsorted(ref_cmf, prob_min)
            if (t_max - t_min) >= 10:
                ref_pmf[0: t_min] = 0.
                ref_pmf[t_max:] = 0.
                ref_pmf_matrix[i] = 0.991 / np.sum(ref_pmf) * ref_pmf
                logging.info(
                    f"Search region for tau[{i}] is restricted to "
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
def full_tau_pretrain_low_tau(parameters, tau_dims):
    """
    Return the probability distribution without cut-off of level 0 to n-1.
    """
    parameters["tau"] = (np.iinfo(np.int32).max,) * len(parameters["protocol"])
    full_result = repeater_sim(parameters, all_level=True)
    ref_pmf_matrix = np.array([result_pair[0] for result_pair in full_result])
    return ref_pmf_matrix[:-1]


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
    # np.random.seed(1)
    ID = log_init("repeater", level=logging.INFO)

    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.98,
        "tau": [(1000000000,)],
        "w_cut": [0.8],
        "t_coh": 400,
        "t_trunc": 3000,
        "cut_type": ["memory_time"],
        "sample_size": 1000000,
        "disc_kind": "both",
        "reuse_sampled_data": False
        }
    # n = 2
    # parameters["p_gen"] = 1-(1-parameters["p_gen"])**n
    # parameters["t_coh"] = parameters["t_coh"]//n

    kwarg_list = create_iter_kwargs(parameters)

    for (timestep, kwarg) in enumerate(kwarg_list):
        # kwarg = copy.deepcopy(kwarg)
        # timestep += 1
        # kwarg["p_gen"] = 1 - (1-kwarg["p_gen"])**(1/timestep)
        # kwarg["t_coh"] = kwarg["t_coh"] * timestep
        # kwarg["t_trunc"] = kwarg["t_trunc"] * timestep

        # logging.info("Level by level optimization\n")
        # best_level_tau = optimization_by_level(kwarg, plot=True)

        logging.info("Uniform tau optimization\n")
        opt = CutoffOptimizer(
            opt_kind="uniform_de", use_tracker=True, adaptive=True,
            tolerance=0., workers=10)
        best_tau = opt.run(kwarg, tau_dims=1)

        # logging.info("Full tau optimization\n")
        # tau_dims = len(parameters["protocol"])
        # opt = CutoffOptimizer(
        #     use_tracker=False, adaptive=True, tolerance=0., workers=10)
        # best_tau = opt.run(kwarg, tau_dims=tau_dims)

        # fig, axs = plt.subplots(2, 2)
        kwarg["tau"] = np.iinfo(np.int).max
        # kwarg["tau"] = best_tau
        pmf, w_func = repeater_sim(parameters=kwarg)
        print("trunc", np.searchsorted(np.cumsum(pmf), 0.99))
        w_func[0] = np.nan
        # plot_algorithm(pmf, w_func, axs=axs)
        # fig.show()
        key_rate = secret_key_rate(pmf, w_func)
        logging.info("Rate without truncation time: {}\n".format(key_rate))
    w_func1 = w_func
    log_finish(ID, parameters, remark="time step")
