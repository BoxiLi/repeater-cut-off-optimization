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
        "t_trunc": 300
        }

    logging.info("Full tau optimization\n")
    opt = CutoffOptimizer(adaptive=True)
    best_cutoff_dict = opt.run(parameters)
    assert(best_cutoff_dict["memory_time"] == (6,))


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

    opt = CutoffOptimizer(adaptive=True, popsize=5)
    best_cutoff_dict = opt.run(parameters)
    assert_allclose(best_cutoff_dict["memory_time"], (18, 30))


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
        opt_kind="uniform_de", adaptive=True)
    best_cutoff_dict = opt.run(parameters)
    assert_allclose(best_cutoff_dict["memory_time"], (45, 45))
