import numpy as np
from numpy.testing import assert_allclose, run_module_suite, assert_
import pytest
import multiprocessing

from repeater_mc import repeater_mc
from repeater_algorithm import repeater_sim, RepeaterChainSimulation
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



swap_only_protocol = {
    "protocol": (0, 0, 0),
    "p_gen": 0.5,
    "p_swap": 0.8,
    "sample_size": 100000,
    "w0": 1.,
    "t_coh": 50,
    "t_trunc": 100
    }

dist_only_protocol = {
    "protocol": (1, 1),
    "p_gen": 0.5,
    "p_swap": 0.8,
    "sample_size": 100000,
    "w0": 1.,
    "t_coh": 20,
    "t_trunc": 100
    }

memory_cutoff_parameters1 = {
    "protocol": (0, 0, 0),
    "p_gen": 0.5,
    "p_swap": 0.8,
    "mt_cut": 5,
    "sample_size": 200000,
    "w0": 1.,
    "t_coh": 30,
    "t_trunc": 100
    }

memory_cutoff_parameters2 = {
    "protocol": (1, 0, 1, 0, 1, 0),
    "p_gen": 0.5,
    "p_swap": 0.8,
    "mt_cut": (3, 6, 10, 14, 25, 100),
    "sample_size": 200000,
    "w0": 1.,
    "t_coh": 30,
    "t_trunc": 150
    }

fidelity_cutoff_parameters = {
    "protocol": (1, 0),
    "p_gen": 0.5,
    "p_swap": 0.8,
    "w0": 0.99,
    "t_coh": 50,
    "t_trunc": 100,
    "cut_type": "fidelity",
    "w_cut": 0.9,
    "sample_size": 1000000,
    }

runtime_cutoff_parameters = {
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


_convolution_simulator = RepeaterChainSimulation()
_convolution_simulator.use_fft = False

_fft_simulator = RepeaterChainSimulation()
_fft_simulator.use_fft = True

_gpu_simulator = RepeaterChainSimulation()
_gpu_simulator.use_fft = True
_gpu_simulator.use_gpu = True
_gpu_simulator.gpu_threshold = 1


def default_solution(parameters):
    simulator = RepeaterChainSimulation()
    simulator.efficient = False
    simulator.use_fft = False
    pmf, w_func = simulator.nested_protocol(parameters)
    return pmf, w_func


@pytest.mark.parametrize("parameters, expect",
    [
        pytest.param(swap_only_protocol, default_solution(swap_only_protocol), id="swap_only"),
        pytest.param(dist_only_protocol, default_solution(dist_only_protocol), id="dist_only"),
        pytest.param(memory_cutoff_parameters1, default_solution(memory_cutoff_parameters1), id="swap_memory_cutoff"),
        pytest.param(memory_cutoff_parameters2, default_solution(memory_cutoff_parameters2), id="swap_dist_memory_cutoff"),
    ])
@pytest.mark.parametrize(("simulator", "efficient"),
    [
        pytest.param(_convolution_simulator, False, id="compartible-conv"),
        pytest.param(_convolution_simulator, True, id="efficient-conv"),
        pytest.param(_fft_simulator, True, id="compartible-fft"),
        pytest.param(_fft_simulator, False, id="efficient-fft"),
        pytest.param(_gpu_simulator, True, id="compartible-gpu"),
    ])
def test_algorithm(parameters, expect, simulator, efficient):
    default_pmf, default_w_func = expect
    simulator.efficient = True
    pmf, w_func = simulator.nested_protocol(parameters)
    cdf = np.cumsum(pmf)
    start_pos = next(x[0] for x in enumerate(cdf) if x[1] > 1.0e-2)
    end_pos = np.searchsorted(cdf, 0.99)
    assert_allclose(pmf[start_pos: end_pos], default_pmf[start_pos: end_pos], rtol=1.0e-7)
    assert_allclose(w_func[start_pos: end_pos], default_w_func[start_pos: end_pos], rtol=1.0e-7)


@pytest.mark.parametrize(
    "parameters, begin, end, rtol_t, rtol_w",
    [
        pytest.param(swap_only_protocol, 5, 12, 0.03, 0.02, id="Swap only protocol"),
        pytest.param(dist_only_protocol, 3, 12, 0.03, 0.02, id="Dist only protocol"),
        pytest.param(memory_cutoff_parameters1, 2, 17, 0.04, 0.02, id="Swap with memory cutoff"),
        pytest.param(memory_cutoff_parameters2, 10, 40, 0.03, 0.02, id="Mixed protocol with memory cutoff"),
        pytest.param(fidelity_cutoff_parameters, 2, 12, 0.03, 0.01, id="Fidelity cutoff"),
        pytest.param(runtime_cutoff_parameters, 2, 12, 0.03, 0.01, id="Runtime cutoff"),
    ])
def test_against_MC(parameters, begin, end, rtol_t, rtol_w):
    simulator = RepeaterChainSimulation()
    pmf, w_func = simulator.nested_protocol(parameters)
    cdf = np.cumsum(pmf)

    pmf_sim, w_func_sim = repeater_mc(parameters, return_pmf=True)
    cdf_sim = np.cumsum(pmf_sim)

    assert_allclose(cdf[begin: end], cdf_sim[begin: end], rtol=rtol_t)
    assert_allclose(w_func[begin: end], w_func_sim[begin: end], rtol=rtol_w)


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


# # test using cutoff as key