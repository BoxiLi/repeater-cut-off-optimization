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

from protocol_units import join_links_compatible
from protocol_units_efficient import join_links_efficient
from utility_functions import secret_key_rate, ceil
from logging_utilities import log_init, create_iter_kwargs, save_data
from repeater_mc import repeater_mc, plot_mc_simulation


__all__ = ["compute_unit", "plot_algorithm",
           "join_links_compatible", "repeater_sim"]


class RepeaterChainSimulation():
    def __init__(self):
        self._GPU_CONVOLUTION_THRESHOLD = 100000000
        self._FFT_CONVOLUTION_THRESHOLD = 1000
        self.efficient = True

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
        size: int
            The truncation time that determines the number of sums
        shift: int, optional
            For each k the function will be shifted to the right. Using for
            time-out mt_cut.
        first_func: array-like, optional
            The first_function in the convolution. If not given, use func.
            It can be different because the first_func is `P_s` and the `func` P_f.
            It is upper bounded by 1.
        coeffs: array-like optional
            The additional factor when sum over k, default is 1.

        Returns
        -------
        sum_convolved: array-like
            The result of the sum of all convolutions.
        """
        target_size = len(func)
        if first_func is None:
            first_func = func
        
        # determine the required number of convolution
        if shift != 0:
            # mt_cut is added here.
            # because it is a constant, we only need size/mt_cut convolution.
            max_k = int(np.ceil((target_size/shift)))
        else:
            max_k = target_size
        if p_swap is not None:
            pf = np.sum(func) * (1 - p_swap)
        else:
            pf = np.sum(func)
        with np.errstate(divide='ignore'):
            max_k = min(max_k, (-52 - np.log(target_size))/ np.log(pf))
        max_k = int(max_k)

        # initialize the result array
        sum_convolved = np.zeros(target_size)
        if p_swap is not None:
            sum_convolved[:len(first_func)] = p_swap * first_func
        else:
            sum_convolved[:len(first_func)] = first_func

        # decide what convolution to use and prepare the data
        convolved = first_func
        length = len(convolved)
        if length > self._FFT_CONVOLUTION_THRESHOLD:
            shape = 2 * target_size - 1
            # The following is from SciPy, they choose the size to be 2^n,
            # It increases the accuracy.
            shape = 2 ** np.ceil(np.log2(shape)).astype(int)
            if _use_cupy and length > self._GPU_CONVOLUTION_THRESHOLD:
                # transfer the data to GPU
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
            convolved_fourier = fft(convolved, shape)
            func_fourier = fft(func, shape)
        else:
            convolved = np.concatenate([convolved, np.zeros(target_size - len(convolved))])

        # perform convolution
        for k in range(1, max_k):
            zero_terms_size = k*shift
            usefull_terms_size = target_size - zero_terms_size
            if length > self._FFT_CONVOLUTION_THRESHOLD:  # convolution in the fourier space
                convolved_fourier = fft(convolved, shape)
                convolved_fourier *= func_fourier
                convolved = ifft(convolved_fourier, shape)
                convolved = to_real(convolved)
            else:
                convolved = np.convolve(
                    convolved[:usefull_terms_size], func[:usefull_terms_size])[:usefull_terms_size]
            # The first k+1 elements should be 0, but FFT convolution
            # gives a non-zero value of about, e-20. It remains to
            # see wether this will have effect on other elements
            # This is important for the first few value of Werner parameters
            convolved[:k+1] = 0.
            if p_swap is not None:
                coeff = p_swap*(1-p_swap)**(k)
                sum_convolved[zero_terms_size:] += coeff * convolved[:usefull_terms_size]
            else:
                sum_convolved[zero_terms_size:] += convolved[:usefull_terms_size]
        if _use_cupy and length > self._GPU_CONVOLUTION_THRESHOLD:
            sum_convolved = cp.asnumpy(sum_convolved)
        return sum_convolved

    def entanglement_swap(self,
            pmf1, w_func1, pmf2, w_func2, p_swap,
            mt_cut, w_cut, rt_cut, t_coh, cut_type):
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
        size: int
            The truncation time, also the size of the matrix.

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
            shift = mt_cut
        else:
            shift = 0

        # P'_f
        pf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=False,
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
        # P'_s
        ps_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
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
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type,
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

        with np.errstate(divide='ignore'):
            state_out[1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
        return pmf_swap, state_out


    def destillation(self,
            pmf1, w_func1, pmf2, w_func2,
            mt_cut, w_cut, rt_cut, t_coh, cut_type):
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
        size: int
            The truncation time, also the size of the matrix.

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
            shift = mt_cut
        else:
            shift = 0
        # P'_f  cutoff attempt when cutoff fails
        pf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=False,
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type,
            evaluate_func="1", t_coh=t_coh)
        # P'_ss  cutoff attempt when cutoff and dist succeed
        pss_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type,
            evaluate_func="0.5+0.5w1w2", t_coh=t_coh)
        # P_s  dist attempt when dist succeeds
        ps_dist = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=pss_cutoff)
        del pss_cutoff
        # P'_sf  cutoff attempt when cutoff succeeds but dist fails
        psf_cutoff = join_links(
            pmf1, pmf2, w_func1, w_func2, ycut=True,
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type,
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
            mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, cut_type=cut_type,
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

        with np.errstate(divide='ignore'):
            state_out[1:] /= pmf_dist[1:]
        return pmf_dist, state_out


    def compute_unit(self,
            parameters, pmf1, w_func1, pmf2=None, w_func2=None, unit_kind="swap", step_size=1):
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
        if "cut_type" not in parameters:
            cut_type = "memory_time"
        else:
            cut_type = parameters["cut_type"]
        t_coh = parameters.get("t_coh", np.inf)
        mt_cut = parameters.get("mt_cut", np.iinfo(np.int).max)
        w_cut = parameters.get("w_cut", 1.0e-8)  # shouldn't be zero
        rt_cut = parameters.get("rt_cut", np.iinfo(np.int).max)

        if not isinstance(p_gen, float) or not isinstance(p_swap, float):
            raise TypeError("p_gen and p_swap must be a float number.")
        if not np.issubdtype(type(mt_cut), np.integer):
            raise TypeError(f"Memory cut-off must be an integer. not {mt_cut}")
        if not np.issubdtype(type(rt_cut), np.integer):
            raise TypeError(f"Run time cut-off must be an integer. not {rt_cut}")
        if not np.isreal(w0) or w0 < 0. or w0 > 1.:
            raise TypeError(f"Fidelity cut-off must be a real number between 0 and 1.")
        if not np.isreal(t_coh):
            raise TypeError(
                f"The coherence time muzst be a real number, not{t_coh}")
        if not np.isreal(w0) or w0 < 0. or w0 > 1.:
            raise TypeError(f"Invalid Werner parameter w0 = {w0}")

        t_coh = t_coh / step_size
        mt_cut = ceil(mt_cut / step_size)
        rt_cut = ceil(rt_cut / step_size)

        # swap or distillation for next level
        if unit_kind == "swap":
            pmf, w_func = self.entanglement_swap(
                pmf1, w_func1, pmf2, w_func2, p_swap,
                mt_cut, w_cut, rt_cut, t_coh, cut_type=cut_type)
        elif unit_kind == "dist":
            pmf, w_func = self.destillation(
                pmf1, w_func1, pmf2, w_func2,
                mt_cut, w_cut, rt_cut, t_coh, cut_type=cut_type)

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

        return pmf, w_func, step_size


    def run_simulation(self, parameters, all_level=False):
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
            parameters["mt_cut"] = mt_cut[i]
            parameters["w_cut"] = w_cut[i]
            parameters["rt_cut"] = rt_cut[i]
        
            if operation == 0:
                pmf, w_func, step_size = self.compute_unit(
                    parameters, pmf, w_func, unit_kind="swap", step_size=total_step_size)
            elif operation == 1:
                pmf, w_func, step_size = self.compute_unit(
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
    simulator = RepeaterChainSimulation()
    return simulator.compute_unit(
        parameters=parameters, pmf1=pmf1, w_func1=w_func1, pmf2=pmf2, w_func2=w_func2, unit_kind=unit_kind, step_size=step_size)


def repeater_sim(parameters, all_level=False):
    simulator = RepeaterChainSimulation()
    return simulator.run_simulation(parameters=parameters, all_level=all_level)


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
