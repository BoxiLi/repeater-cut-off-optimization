import warnings
from copy import deepcopy

import numba as nb
import numpy as np
from scipy.optimize import curve_fit


@nb.jit(nopython=True)
def pmf_to_cdf(pmf):
    return np.cumsum(pmf)


@nb.jit(nopython=True)
def cdf_to_pmf(cdf):
    pmf = np.zeros(len(cdf))
    for i in range(len(cdf)):
        if i==0:
            pmf[0] = cdf[0]
        else:
            pmf[i] = cdf[i] - cdf[i-1]
    return pmf


def prob_abs_diff_sim(pmf, max_diff):
    """
    Sum up to |X-X'| <= max_diff
    """
    len_pmf = len(pmf)
    pmf2 = np.zeros(len_pmf)
    for i in range(len_pmf):  # numba does not support np.flip
        pmf2[len_pmf - i - 1] = pmf[i]
    diff = np.convolve(pmf, pmf2)
    left_half = (len(diff) - 1) // 2
    abs_diff = diff[left_half:]
    for i in range(0, left_half):
        abs_diff[1 + i] += diff[left_half - i - 1]
    return np.sum(abs_diff[0: max_diff + 1])


def werner_to_fid(werner):
    return (1. + 3. * werner) / 4.


def fid_to_werner(fid):
    return (4 * fid - 1) / 3.


def entropy(x):
    if x==0.:
        return 0.
    elif x==1.:
        return 0.
    return -x * np.log(x) - (1-x) * np.log(1-x)


def distillable_entanglement(w_func):
    f_func = werner_to_fid(w_func)
    f_func[f_func < 0.5] = 0.5
    f_func[f_func == 0.5] = 0.5 + 1.e-7  # avoid log(0)
    return entropy(0.5 + (f_func * (1-f_func))**0.5)


def secret_fraction(w):
    """
    Secret fraction of BB84 protocol with Werner state.
    
    Parameters
    ----------
    w: float
        Werner parameter
    
    Returns
    -------
    secret_fraction: float
        Secret fraction
    """
    return max(1 - 2. * entropy((1.-w)/2.), 0.)


def secret_key_rate(pmf, w_func, extrapolation=False, show_warning=False):
    """
    Use the secret key rate as a merit function.
    It is defined by the multiplication of raw key rate and the
    secret key fraction.
    """
    coverage = np.sum(pmf)
    aver_w = get_mean_werner(pmf, w_func, extrapolation)
    aver_t = get_mean_waiting_time(pmf, extrapolation, show_warning)

    key_rate = 1/aver_t * secret_fraction(aver_w)
    if key_rate < 0.:
        key_rate = 0.
    return key_rate


def get_mean_werner(pmf, w_func, extrapolation=False):
    w_func = np.where(np.isnan(w_func), 0., w_func)
    coverage = np.sum(pmf)
    if not extrapolation or coverage > 1 - 1.e-10 or coverage < 0.99:
        aver_w = np.sum(pmf * w_func) / coverage
    else:
        aver_w = np.sum(pmf * w_func) + w_func[-1] * (1. - coverage)
    return aver_w
    

def get_mean_waiting_time(pmf, extrapolation=False, show_warning=False):
    coverage = np.sum(pmf)
    # if coverage < 0.99, extrapolation may leads to wrong secret key rate
    # if coverage > 1 - 1.e-10, the last few point is close to 0 and therefore the numerical noise dominant.
    if not extrapolation or coverage > 1 - 1.e-10 or coverage < 0.99:
        return (1 - coverage) / coverage * len(pmf) + np.sum(pmf * np.arange(len(pmf)))/coverage
    else:
        # use an exponential function to estimate the distribution
        t_trunc = len(pmf)
        sample_start = t_trunc - 10
        sample_end = t_trunc

        def func(x, a, b):
            c_0 = pmf[-1]
            d_0 = (pmf[-2] - pmf[-1]) / c_0
            with warnings.catch_warnings(record=True) as w:
                # ignore np.exp overflow warning
                warnings.simplefilter("ignore")
                return c_0 * np.exp(-d_0 * a * x + b)
        with warnings.catch_warnings() as w:
            if not show_warning:
                warnings.simplefilter("ignore")
            par, cov = curve_fit(func,
                             np.arange(sample_start, sample_end),
                             pmf[sample_start: sample_end])
        a = par[0]
        b = par[1]
        c_0 = pmf[-1]
        d_0 = (pmf[-2]-pmf[-1])/c_0
        rest = c_0 * np.exp(-a * d_0 * t_trunc + b) / a / d_0
        aver_t_rest = (a * d_0 * t_trunc + 1.) / a / d_0 * rest
        return np.sum(pmf * np.arange(t_trunc)) + aver_t_rest


def create_cutoff_dict(cutoff_list, cut_types, parameters, ref_pmf_matrix=None):
    cutoff_list = np.asarray(cutoff_list)
    if isinstance(cut_types, str):
        cut_types = [cut_types]
    num_cut_types = len(cut_types)
    cutoff_mat = cutoff_list.reshape((num_cut_types, len(cutoff_list)//num_cut_types))
    cutoff_dict = {cut_types[i]: cutoff_mat[i] for i in range(num_cut_types)}
    for cut_type in cut_types:
        cutoff = cutoff_dict[cut_type]
        if cut_type in ["memory_time", "run_time"]:
            if ref_pmf_matrix is not None:
                if all(cutoff < 0.) or all(cutoff > 1.):
                    raise ValueError(
                        "A reference pmf is given, but cutoff is not a probability")
                # if len(cutoff) != len(ref_pmf_matrix):
                #     raise ValueError(
                #         "The reference probability matrix must have "
                #         "the same length as the input cutoff. However\n "
                #         "len(cutoff)={}\n len(ref_pmf_matrix)={}\n".format(
                #             len(cutoff), len(ref_pmf_matrix)))
                cutoff_pos = cutoff
                cutoff = np.empty(cutoff_pos.shape, dtype=np.int)
                for i in range(len(cutoff)):
                    cutoff[i] = np.searchsorted(np.cumsum(ref_pmf_matrix[i]), cutoff_pos[i])
            # make sure runtime cutoff is in the increasing order
            if cut_type in ["run_time"]:
                for i in range(1, len(cutoff)):
                    cutoff[i] = min(cutoff[i] + cutoff[i-1], parameters["t_trunc"])
            cutoff = cutoff.astype(np.int)
        if len(cutoff) == 1 and len(parameters["protocol"]) != 1:
            cutoff = np.repeat(cutoff, len(parameters["protocol"]))
        cutoff_dict[cut_type] = cutoff
    return cutoff_dict


def ceil(float_number):
    return int(np.ceil(float_number))


def find_heading_zeros_num(array):
    heading_zeros_num = 0
    for i in range(len(array)):
        if array[i] == 0.:
            heading_zeros_num += 1
        else:
            break
    return heading_zeros_num


def werner_to_matrix(w):
    if np.isscalar(w):
        identity = np.eye(4)/4
        phi = np.outer([0,1,1,0], [0,1,1,0])/2
        return w * phi + (1-w) * identity
    else:
        result = np.empty((len(w), 4, 4), dtype=float)
        for i in range(len(w)):
            result[i] = werner_to_matrix(w[i])
        return result


def matrix_to_werner(mat):
    if mat.shape == (4, 4):
        return (mat[1, 1] - mat[0, 0]) * 2
    else:
        result = np.empty(len(mat), dtype=float)
        for i in range(len(mat)):
            result[i] = matrix_to_werner(mat[i])
        return result

def get_fidelity(state, ket):
    if state.shape == (4, 4):
        return np.sqrt(np.real(np.transpose(ket) @ state @ ket))
    else:
        result = np.empty(len(state), dtype=float)
        for i in range(len(state)):
            result[i] = matrix_to_werner(state[i])
        return result
