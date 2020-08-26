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
    _cupy_exist = True
except (ImportError, ModuleNotFoundError):
    _cupy_exist = False

from protocol_units import join_links_compatible
from protocol_units_efficient import join_links_efficient
from utility_functions import secret_key_rate, ceil, werner_to_fid, find_heading_zeros_num, matrix_to_werner, werner_to_matrix, get_fidelity
from logging_utilities import log_init, create_iter_kwargs, save_data
from repeater_mc import repeater_mc, plot_mc_simulation
from nv_state import nv_elementary_link


__all__ = ["RepeaterChainSimulation", "compute_unit", "plot_algorithm",
           "join_links_compatible", "repeater_sim"]


class RepeaterChainSimulation():
    def __init__(self):
        self.use_fft = True
        self.use_gpu = False
        self.gpu_threshold = 1000000
        self.efficient = True
        self.zero_padding_size = 1

    def iterative_convolution(self,
            func, shift=0, first_func=None, p_swap=None):
        """
        Calculate the convolution iteratively:
        first_func * func * func * ... * func
        It returns the sum of all iterative convolution:
        first_func + first_func * func + first_func * func * func ...

        Parameters
        ----------
        func: array-like
            The function to be convolved in array form.
            It is always a probability distribution.
        shift: int, optional
            For each k the function will be shifted to the right. Using for
            time-out mt_cut.
        first_func: array-like, optional
            The first_function in the convolution. If not given, use func.
            It can be different because the first_func is
            `P_s` and the `func` P_f.
            It is upper bounded by 1.
            It can be a probability, or an array of states.
        p_swap: float, optimal
            Entanglement swap success probability.

        Returns
        -------
        sum_convolved: array-like
            The result of the sum of all convolutions.
        """
        if first_func is None or len(first_func.shape) == 1:
            is_dm = False
        else:
            is_dm = True

        trunc = len(func)

        # determine the required number of convolution
        if shift != 0:
            # cut-off is added here.
            # because it is a constant, we only need size/mt_cut convolution.
            max_k = int(np.ceil((trunc/shift)))
        else:
            max_k = trunc
        if p_swap is not None:
            pf = np.sum(func) * (1 - p_swap)
        else:
            pf = np.sum(func)
        with np.errstate(divide='ignore'):
            max_k = min(max_k, (-52 - np.log(trunc))/ np.log(pf))
        max_k = int(max_k)

        # Transpose the array of state to the shape (1,1,trunc)
        # if werner or shape (4,4,trunc) if density matrix
        if first_func is None:
            first_func = func
        if not is_dm:
            first_func = first_func.reshape((trunc, 1, 1))
        first_func = np.transpose(first_func, (1, 2, 0))

        # Convolution
        result = np.empty(first_func.shape, first_func.dtype)
        for i in range(first_func.shape[0]):
            for j in range(first_func.shape[1]):
                result[i][j] = self.iterative_convolution_helper(
                    func, first_func[i][j], trunc, shift, p_swap, max_k)

        # Permute the indices back
        result = np.transpose(result, (2, 0, 1))
        if not is_dm:
            result = result.reshape(trunc)

        return result

    def iterative_convolution_helper(
            self, func, first_func, trunc, shift, p_swap, max_k):
        # initialize the result array
        sum_convolved = np.zeros(trunc)
        if p_swap is not None:
            sum_convolved[:len(first_func)] = p_swap * first_func
        else:
            sum_convolved[:len(first_func)] = first_func

        if shift <= trunc:
            zero_state = np.zeros(shift)
            func = np.concatenate([zero_state, func])[:trunc]

        # decide what convolution to use and prepare the data
        convolved = first_func
        if self.use_fft: # Use geometric sum in Fourier space
            shape = 2 * trunc - 1
            # The following is from SciPy, they choose the size to be 2^n,
            # It increases the accuracy.
            shape = 2 ** np.ceil(np.log2(shape)).astype(int)
            if self.use_gpu and shape > self.gpu_threshold:
                # transfer the data to GPU
                sum_convolved = cp.asarray(sum_convolved)
                convolved = cp.asarray(convolved)
                func = cp.asarray(func)
            if self.use_gpu and shape > self.gpu_threshold:
                # use CuPy fft
                ifft = cp.fft.ifft
                fft = cp.fft.fft
                to_real = cp.real
            else:
                # use NumPy fft
                ifft = np.fft.ifft
                fft = np.fft.fft
                to_real = np.real

            convolved_fourier = fft(convolved, shape)
            func_fourier = fft(func, shape)

            if p_swap is not None:
                result= ifft(
                    p_swap*convolved_fourier / (1 - (1-p_swap) * func_fourier))
            else:
                result= ifft(convolved_fourier / (1 - func_fourier))
            result = to_real(result[:trunc])
            if self.use_gpu and shape > self.gpu_threshold:
                result = cp.asnumpy(result)

        else:  # Use exact convolution
            zero_state = np.zeros(trunc - len(convolved))
            convolved = np.concatenate([convolved, zero_state])
            for k in range(1, max_k):
                convolved = np.convolve(convolved[:trunc], func[:trunc])
                if p_swap is not None:
                    coeff = p_swap*(1-p_swap)**(k)
                    sum_convolved += coeff * convolved[:trunc]
                else:
                    sum_convolved += convolved[:trunc]
            result = sum_convolved
        return result

    def entanglement_swap(self,
            pmf1, w_func1, pmf2, w_func2, p_swap,
            cutoff, t_coh, cut_type):
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
        cutoff: int or float
            The memory time cut-off, werner parameter cut-off, or 
            run time cut-off.
        t_coh: int
            The coherence time.
        cut_type: str
            `memory_time`, `fidliety` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the entanglement swap.
        w_func: array-like 1-D
            The Werner parameter as function of T of the entanglement swap.
        """
        if self.efficient and cut_type == "memory_time":
            join_links = join_links_efficient
        else:
            join_links = join_links_compatible
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0

        # P'_f
        pf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=False,
            cutoff=cutoff, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
        # P'_s
        ps_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
        # P_f or P_s (Differs only by a constant p_swap)
        pmf_cutoff = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=ps_cutoff)
        del ps_cutoff
        # Pr(Tout = t)
        pmf_swap = self.iterative_convolution(
            pmf_cutoff, shift=0, p_swap=p_swap)

        # Wsuc * P_s
        state_suc = join_links(
            pmf1, pmf2, w_func1=w_func1, w_func2=w_func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            t_coh=t_coh, evaluate_func="w1w2")
        # Wprep * Pr(Tout = t)
        state_prep = self.iterative_convolution(
            pf_cutoff,
            shift=shift, first_func=state_suc)
        del pf_cutoff, state_suc
        # Wout * Pr(Tout = t)
        state_out = self.iterative_convolution(
            pmf_cutoff, shift=0,
            first_func=state_prep, p_swap=p_swap)
        del pmf_cutoff

        with np.errstate(divide='ignore', invalid='ignore'):
            state_out[1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
            state_out = np.where(np.isnan(state_out), 1., state_out)

        return pmf_swap, state_out

    def destillation(self,
            pmf1, w_func1, pmf2, w_func2,
            cutoff, t_coh, cut_type):
        """
        Calculate the waiting time and average Werner parameter
        with time-out for the distillation.

        Parameters
        ----------
        pmf1, pmf2: array-like 1-D
            The waiting time distribution of the two input links.
        w_func1, w_func2: array-like 1-D
            The Werner parameter as function of T of the two input links.
        cutoff: int or float
            The memory time cut-off, werner parameter cut-off, or 
            run time cut-off.
        t_coh: int
            The coherence time.
        cut_type: str
            `memory_time`, `fidliety` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the distillation.
        w_func: array-like 1-D
            The Werner parameter as function of T of the distillation.
        """
        if self.efficient and cut_type == "memory_time":
            join_links = join_links_efficient
        else:
            join_links = join_links_compatible
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0
        # P'_f  cutoff attempt when cutoff fails
        pf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=False,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="1", t_coh=t_coh)
        # P'_ss  cutoff attempt when cutoff and dist succeed
        pss_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="0.5+0.5w1w2", t_coh=t_coh)
        # P_s  dist attempt when dist succeeds
        ps_dist = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=pss_cutoff)
        del pss_cutoff
        # P'_sf  cutoff attempt when cutoff succeeds but dist fails
        psf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="0.5-0.5w1w2", t_coh=t_coh)
        # P_f  dist attempt when dist fails
        pf_dist = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=psf_cutoff)
        del psf_cutoff
        # Pr(Tout = t)
        pmf_dist = self.iterative_convolution(
            pf_dist, shift=0,
            first_func=ps_dist)
        del ps_dist

        # Wsuc * P'_ss
        state_suc = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="w1+w2+4w1w2", t_coh=t_coh)
        # Wprep * P_s
        state_prep = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=state_suc)
        del pf_cutoff, state_suc
        # Wout * Pr(Tout = t)
        state_out = self.iterative_convolution(
            pf_dist, shift=0,
            first_func=state_prep)
        del pf_dist, state_prep

        with np.errstate(divide='ignore', invalid='ignore'):
            state_out[1:] /= pmf_dist[1:]
            state_out = np.where(np.isnan(state_out), 1., state_out)
        return pmf_dist, state_out


    def compute_unit(self,
            parameters, pmf1, w_func1, pmf2=None, w_func2=None,
            unit_kind="swap", step_size=1):
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
        w0 = parameters["w0"]
        t_coh = parameters.get("t_coh", np.inf)
        cut_type = parameters.get("cut_type", "memory_time")
        if "cutoff" in parameters.keys():
            cutoff = parameters["cutoff"]
        elif cut_type == "memory_time":
            cutoff = parameters.get("mt_cut", np.iinfo(np.int).max)
        elif cut_type == "fidelity":
            cutoff = parameters.get("w_cut", 1.0e-16)  # shouldn't be zero
            if cutoff == 0.:
                cutoff = 1.0e-16
        elif cut_type == "run_time":
            cutoff = parameters.get("rt_cut", np.iinfo(np.int).max)
        else:
            cutoff = np.iinfo(np.int).max

        # type check
        if not isinstance(p_gen, float) or not isinstance(p_swap, float):
            raise TypeError("p_gen and p_swap must be a float number.")
        if cut_type in ("memory_time", "run_time") and not np.issubdtype(type(cutoff), np.integer):
            raise TypeError(f"Time cut-off must be an integer. not {cutoff}")
        if cut_type == "fidelity" and not (cutoff >= 0. or cutoff < 1.):
            raise TypeError(f"Fidelity cut-off must be a real number between 0 and 1.")
        if not np.isreal(t_coh):
            raise TypeError(
                f"The coherence time must be a real number, not{t_coh}")
        if not np.isreal(w0) or w0 < 0. or w0 > 1.:
            raise TypeError(f"Invalid Werner parameter w0 = {w0}")

        # swap or distillation for next level
        if unit_kind == "swap":
            pmf, w_func = self.entanglement_swap(
                pmf1, w_func1, pmf2, w_func2, p_swap,
                cutoff=cutoff, t_coh=t_coh, cut_type=cut_type)
        elif unit_kind == "dist":
            pmf, w_func = self.destillation(
                pmf1, w_func1, pmf2, w_func2,
                cutoff=cutoff, t_coh=t_coh, cut_type=cut_type)

        # erase ridiculous Werner parameters,
        # it can happen when the probability is too small ~1.0e-20.
        w_func = np.where(np.isnan(w_func), 1., w_func)
        w_func[w_func > 1.0] = 1.0
        w_func[w_func < 0.] = 0.

        # check probability coverage
        coverage = np.sum(pmf)
        if coverage < 0.99:
            logging.warning(
                "The truncation time only covers {:.2f}% of the distribution, "
                "please increase t_trunc.\n".format(
                    coverage*100))
        
        return pmf, w_func


    def nested_protocol(self, parameters, all_level=False):
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
        w0 = parameters["w0"]
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
        if "cutoff" in parameters:
            cutoff = parameters["cutoff"]
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

        t_trunc = parameters["t_trunc"]

        # elementary link
        t_list = np.arange(1, t_trunc)
        pmf = p_gen * (1 - p_gen)**(t_list - 1)
        pmf = np.concatenate((np.array([0.]), pmf))
        w_func = np.array([w0] * t_trunc)
        if all_level:
            full_result = [(pmf, w_func)]

        total_step_size = 1
        # protocol unit level by level
        for i, operation in enumerate(protocol):
            if "cutoff" in parameters and isinstance(cutoff, Iterable):
                parameters["cutoff"] = cutoff[i]
            parameters["mt_cut"] = mt_cut[i]
            parameters["w_cut"] = w_cut[i]
            parameters["rt_cut"] = rt_cut[i]
        
            if operation == 0:
                pmf, w_func = self.compute_unit(
                    parameters, pmf, w_func, unit_kind="swap", step_size=total_step_size)
            elif operation == 1:
                pmf, w_func = self.compute_unit(
                    parameters, pmf, w_func, unit_kind="dist", step_size=total_step_size)
            if all_level:
                full_result.append((pmf, w_func))

        final_pmf = pmf
        final_w_func = w_func
        if all_level:
            return full_result
        else:
            return final_pmf, final_w_func


def compute_unit(
        parameters, pmf1, w_func1, pmf2=None, w_func2=None,
        unit_kind="swap", step_size=1):
    """
    Functional warpper for compute_unit
    """
    simulator = RepeaterChainSimulation()
    return simulator.compute_unit(
        parameters=parameters, pmf1=pmf1, w_func1=w_func1, pmf2=pmf2, w_func2=w_func2, unit_kind=unit_kind, step_size=step_size)


def repeater_sim(parameters, all_level=False):
    """
    Functional warpper for nested_protocol
    """
    simulator = RepeaterChainSimulation()
    return simulator.nested_protocol(parameters=parameters, all_level=all_level)


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


if __name__ == "__main__":
    # set parameters
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.01,
        "p_swap": 0.5,
        # "cutoff": (175, 319, 553),
        "cutoff": (176, 320, 554),
        "w0": 0.98,
        "t_coh": 4000,
        "t_trunc": 80000,
        }


    ID = log_init("tau_opt", level=logging.INFO)
    fig, axs = plt.subplots(2, 2, dpi=150)
    kwarg_list = create_iter_kwargs(parameters)

    # # simulation part
    # t_sample_list = []
    # w_sample_list = []

    # for kwarg in kwarg_list:
    #     start = time.time()
    #     print("Sample parameters:")
    #     print(kwarg)
    #     t_samples_level, w_samples_level = repeater_mc(kwarg)
    #     t_sample_list.append(t_samples_level)
    #     w_sample_list.append(w_samples_level)
    #     end = time.time()
    #     print("MC Simulation elapse time\n", end-start)
    #     print()
    # save_data(id, data=[t_sample_list, w_sample_list])

    # plot_mc_simulation(
    #     [t_sample_list, w_sample_list], axs, t_trunc=None,
    #     parameters=parameters, bin_width=1)

    # exact
    for kwarg in kwarg_list:
        # n = 10
        # kwarg = deepcopy(kwarg)
        # tau = np.asarray(kwarg["tau"])
        # kwarg["tau"] = tuple(tau * n)
        # kwarg["p_gen"] = 1 - (1-kwarg["p_gen"])**(1/n)
        # kwarg["t_coh"] = kwarg["t_coh"] * n
        # kwarg["t_trunc"] = kwarg["t_trunc"] * n
        # print(kwarg)
        start = time.time()
        simulator = RepeaterChainSimulation()
        pmf, w_func = simulator.nested_protocol(parameters=kwarg)
        end = time.time()
        print("average waiting time", np.sum(pmf * np.arange(len(pmf))))
        print("average w_func", np.sum(pmf * w_func))
        t = 0
        while(pmf[t] < 1.0e-17):
            w_func[t] = np.nan
            t += 1
        print("Deterministic elapse time\n", end-start)
        print()
        plot_algorithm(pmf, w_func, axs, t_trunc=None)
        print("coverage", sum(pmf))
        print("secret without extrap", secret_key_rate(pmf, w_func, False))
        # print("secret with extrap", secret_key_rate(pmf, w_func, True))
        print()

    # plot setup
    legend = None
    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
    plt.show()
    input()