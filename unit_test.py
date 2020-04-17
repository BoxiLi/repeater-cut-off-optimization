import numpy as np
from numpy.testing import assert_allclose, run_module_suite, assert_
import pytest
import multiprocessing

from repeater_mc import repeater_mc
from repeater_algorithm import repeater_sim, join_links
from logging_utilities import *
from utility_functions import *
from optimize_cutoff import CutoffOptimizer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.skip(reason="Outdated")
def test_fail_success_prob():
    """
    Test for the join_links probability
    """
    @nb.jit(nopython=True, error_model="numpy")
    def min_wrap(t1, t2):
        return min(t1, t2)

    pmf = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    w_func = np.random.random(len(pmf))
    t_trunc = len(pmf)

    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=0, ycut=False, waiting=time_cut_off_waiting)[4], 0.)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=0, ycut=False, waiting=time_cut_off_waiting)[0], 0.18)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=1, ycut=False, waiting=time_cut_off_waiting)[3], 0.)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=1, ycut=False, waiting=time_cut_off_waiting)[4], 0.)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=1, ycut=False, waiting=time_cut_off_waiting)[2], 0.1)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=1, ycut=False, waiting=time_cut_off_waiting)[0], 0.12)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=4, ycut=False, waiting=time_cut_off_waiting)[0], 0.)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=5, ycut=False, waiting=time_cut_off_waiting)[4], 0.)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func,
            tcut=0, ycut=False, waiting=time_cut_off_waiting)[3], 0.075)

    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=0, ycut=True)[3], 0.0225)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=0, ycut=True)[0], 0.01)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=0, ycut=True)[4], 0.0625)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=1, ycut=True)[0], 0.01)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=1, ycut=True)[2], 0.16)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=1, ycut=True)[3], 0.0825)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=1, ycut=True)[4], 0.1375)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=4, ycut=True)[0], 0.01)
    assert_allclose(join_links(
            pmf, pmf, w_func, w_func, tcut=5, ycut=True)[4], 0.4375)


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
        "disc_kind": "both",
        "reuse_sampled_data": False,
        "t_trunc": 100
        }
    t_trunc = 100
    pmf, w_func = repeater_sim(parameters)
    cdf = np.cumsum(pmf)

    kwarg_list = create_iter_kwargs(parameters)
    pmf_sim, w_func_sim = repeater_mc(kwarg_list[0], return_pmf=True)

    assert_allclose(pmf[2:17], pmf_sim[2:17], rtol=0.05)
    assert_allclose(w_func[2:17], w_func_sim[2:17], rtol=0.01)


def test_deterministic_dist():
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


def test_opt_adaptive_trunc():
    """
    t_trunc should be increased
    """
    np.random.seed(1)
    parameters = {
        "protocol": (0, ),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.99,
        "t_coh": 30,
        "t_trunc": 100
        }

    logging.info("Full tau optimization\n")
    tau_dims = len(parameters["protocol"])
    opt = CutoffOptimizer(adaptive=True, tolerance=0)
    best_tau = opt.run(parameters, tau_dims=tau_dims)
    assert(best_tau == (6,))


def test_opt_adaptive_search_range():
    """
    The search range should be restricted
    """
    np.random.seed(3)
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.2,
        "p_swap": 0.6,
        "w0": 0.95,
        "t_coh": 300,
        "t_trunc": 400
        }

    tau_dims = len(parameters["protocol"])
    opt = CutoffOptimizer(adaptive=True, tolerance=0, popsize=5)
    best_tau = opt.run(parameters, tau_dims=tau_dims)
    assert(best_tau == (18, 30))


def test_opt_uniform():
    np.random.seed(0)
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.1,
        "p_swap": 0.8,
        "t_trunc": 500,
        "w0": 0.99,
        "t_coh": 400,
        }

    opt = CutoffOptimizer(
        opt_kind="uniform_de", use_tracker=True, adaptive=True, tolerance=0)
    best_tau = opt.run(parameters, tau_dims=1)
    assert(best_tau == (45, 45))
