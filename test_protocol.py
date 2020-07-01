import numpy as np
from numpy.testing import assert_allclose, run_module_suite, assert_
import pytest
import multiprocessing

from repeater_mc import repeater_mc
from repeater_algorithm import repeater_sim
from protocol_units import *
from logging_utilities import *
from utility_functions import *
from optimize_cutoff import CutoffOptimizer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_fidelity_cut_off_function():
    w_cut = 0.95
    t_coh = 50
    assert(fidelity_cut_off(1, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(2, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (3, True))
    assert(fidelity_cut_off(4, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (4, True))
    assert(fidelity_cut_off(3, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (3, True))
    assert(fidelity_cut_off(6, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (4, False))
    assert(fidelity_cut_off(1, 4, 0.98, 0.94, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(1, 2, 0.98, 0.94, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(4, 4, 0.95, 0.95, w_cut=w_cut, t_coh=t_coh) == (4, True))
    assert(fidelity_cut_off(4, 4, 0.95, 0.9499, w_cut=w_cut, t_coh=t_coh) == (4, False))


def test_cutoff_dict_generation():    
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "w0": 1.0,
        "t_coh": 20,
        "t_trunc": 100,
        "cut_type": "fidelity",
        }

    cutoff_dict = create_cutoff_dict((100, 120, 0.5, 0.4), ["memory_time", "fidelity"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 120]))
    assert_allclose(cutoff_dict["fidelity"], np.array([0.5, 0.4]))

    cutoff_dict = create_cutoff_dict((0.8, ), ["fidelity"], parameters)
    assert_allclose(cutoff_dict["fidelity"], np.array([0.8, 0.8]))

    cutoff_dict = create_cutoff_dict((100, ), ["memory_time"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 100]))

    cutoff_dict = create_cutoff_dict((100, 0.5), ["memory_time", "fidelity"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 100]))
    assert_allclose(cutoff_dict["fidelity"], np.array([0.5, 0.5]))

def test_deterministic_swap():
    """
    Test for the deterministic algorithm with swap and distillation.
    """
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "tau": 5,
        "sample_size": 200000,
        "w0": 1.,
        "t_coh": 30,
        "t_trunc": 100
        }
    t_trunc = 100
    pmf, w_func = repeater_sim(parameters)
    cdf = np.cumsum(pmf)

    kwarg_list = create_iter_kwargs(parameters)
    pmf_sim, w_func_sim = repeater_mc(kwarg_list[0], return_pmf=True)

    assert_allclose(pmf[2:17], pmf_sim[2:17], rtol=0.05)
    assert_allclose(w_func[2:17], w_func_sim[2:17], rtol=0.01)


def test_dist():
    """
    Test for the deterministic algorithm with swap,
    distillation and time-out.
    """
    # set parameters
    parameters = {
        "protocol": (1, 0, 1, 0, 1, 0),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "tau": (3, 6, 10, 14, 25, 100),
        "sample_size": 200000,
        "w0": 1.,
        "t_coh": 30,
        "disc_kind": "both",
        "reuse_sampled_data": False,
        "t_trunc": 100
        }

    t_trunc = 100
    pmf, w_func = repeater_sim(parameters)
    cdf = np.cumsum(pmf)

    kwarg_list = create_iter_kwargs(parameters)
    pmf_sim, w_func_sim = repeater_mc(kwarg_list[0], return_pmf=True)

    assert_allclose(pmf[10:40], pmf_sim[10:40], rtol=0.08)
    assert_allclose(w_func[10:40], w_func_sim[10:40], rtol=0.02)


def test_fidelity_cut_off():
    parameters = {
        "protocol": (0, ),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "w0": 0.99,
        "t_coh": 50,
        "t_trunc": 100,
        "cut_type": "fidelity",
        "w_cut": 0.9,
        "sample_size": 1000000,
        }

    t_trunc = 100
    pmf, w_func = repeater_sim(parameters)
    cdf = np.cumsum(pmf)

    kwarg_list = create_iter_kwargs(parameters)
    pmf_sim, w_func_sim = repeater_mc(kwarg_list[0], return_pmf=True)
    assert_allclose(pmf[1:12], pmf_sim[1:12], rtol=0.03)
    assert_allclose(w_func[1:12], w_func_sim[1:12], rtol=0.03)


def test_runtime_cut_off():
    parameters = {
        "protocol": (1, 0),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "w0": 0.99,
        "t_coh": 50,
        "t_trunc": 100,
        "cut_type": "run_time",
        "rt_cut": (5, 10),
        "sample_size": 1000000,
        }
    t_trunc = 100
    pmf, w_func = repeater_sim(parameters)
    cdf = np.cumsum(pmf)

    kwarg_list = create_iter_kwargs(parameters)
    pmf_sim, w_func_sim = repeater_mc(kwarg_list[0], return_pmf=True)
    assert_allclose(pmf[2:13], pmf_sim[2:13], rtol=0.04)
    assert_allclose(w_func[2:15], w_func_sim[2:15], rtol=0.03)


@pytest.mark.skip(reason="Used only locally")
@pytest.mark.filterwarnings("ignore:Record with ID")
def test_record():
    parameters1 = {'ID': 'test1', 'remark': 'test_log'}
    parameters2 = {'ID': 'test2', 'remark': 'test_log'}
    assert(find_record_id("test1") == parameters1)
    assert(find_record_id("test2") == parameters2)
    assert(len(find_record_patterns({'remark': "test_log"})) == 2)
    assert(find_record_id("does not exist") is None)
    assert(find_record_patterns({'ID': "does not exist"}) == [])


def test_secret_key_rate():
    """
    Test secret key rate with exponential extrapolation.
    """
    parameters = {
        "protocol": (0, ),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.99,
        "tau": 5,
        "t_coh": 30,
        "t_trunc": 236
        }
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func, extrapolation=True)
    assert_allclose(key_rate, 0.0148367, rtol=1.e-3)


# test using cutoff as key
# test efficient version
# test version with no cutoff
# test withouf fft and with fft
# test with gpu