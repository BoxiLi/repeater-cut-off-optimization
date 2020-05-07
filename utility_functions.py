import numba as nb
import numpy as np
import warnings
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
try:
    import cupy as cp
    _use_cupy = True
except (ImportError, ModuleNotFoundError):
    _use_cupy = False

from logging_utilities import mytimeit

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


def werner_to_fid(werner):
    return (1. + 3. * werner) / 4.


def fid_to_werner(fid):
    return (4 * fid - 1) / 3.


def entropy(x):
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
    return 1 - 2. * entropy((1.-w)/2.)


def secret_key_rate(pmf, w_func, extrapolation=False, show_warning=False):
    """
    Use the secret key rate as a merit function.
    It is defined by the multiplication of raw key rate and the
    secret key fraction.
    """
    w_func = np.where(np.isnan(w_func), 0., w_func)
    coverage = np.sum(pmf)
    if not extrapolation or coverage > 1 - 1.e-10 or coverage < 0.99:
        aver_w = np.sum(pmf * w_func) / coverage
    else:
        aver_w = np.sum(pmf * w_func) + w_func[-1] * (1. - coverage)

    aver_t = get_mean_waiting_time(pmf, extrapolation, show_warning)

    key_rate = 1/aver_t * secret_fraction(aver_w)
    if key_rate < 0.:
        key_rate = 0.
    return key_rate


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
