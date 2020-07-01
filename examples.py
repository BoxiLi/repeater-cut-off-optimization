import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from repeater_algorithm import repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)
from utility_functions import secret_key_rate


def single_parameter_simulation():
    """
    This example is a simplified version of fig.4 from the paper.
    It calculates the waiting time distribution and the Werner parameter
    with the algorithm shown in the paper.
    A Monte Carlo algorithm is used for comparison.
    It will take about one minutes on a i7 8700 CPU.
    """
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 3-level swap,
        # spanning over 9 nodes (i.e. 8 segments)
        "protocol": (0, 0, 0),
        # success probability of entanglement generation
        "p_gen": 0.1,
        # success probability of entanglement swap
        "p_swap": 0.5,
        # initial Werner parameter
        "w0": 0.98,
        # memory cut-off time
        "tau": (16, 31, 55),
        # the memory coherence time
        "t_coh": 400,
        # truncation time for the repeater scheme
        "t_trunc": 3000,
        # the type of cut-off
        "cut_type": "memory_time",
        # the sample size for the MC algorithm
        "sample_size": 1000000,
        }
    # initialize the logging system
    log_init("sim", level=logging.INFO)
    fig, axs = plt.subplots(2, 2)

    # Monte Carlo simulation
    print("Monte Carlo simulation")
    t_sample_list = []
    w_sample_list = []
    start = time.time()
    # Run the MC simulation
    t_samples, w_samples = repeater_mc(parameters)
    t_sample_list.append(t_samples)
    w_sample_list.append(w_samples)
    end = time.time()
    print("Elapse time\n", end-start)
    print()
    plot_mc_simulation(
        [t_sample_list, w_sample_list], axs,
        parameters=parameters, bin_width=1, t_trunc=2000)

    # Algorithm presented in the paper
    print("Deterministic algorithm")
    start = time.time()
    # Run the calculation
    pmf, w_func = repeater_sim(parameters)
    end = time.time()
    t = 0
    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    while(pmf[t] < 1.0e-17):
        w_func[t] = np.nan
        t += 1
    print("Elapse time\n", end-start)
    print()
    plot_algorithm(pmf, w_func, axs, t_trunc=2000)
    print("secret key rate", secret_key_rate(pmf, w_func))

    # plot
    legend = None
    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
    fig.show()
    input()


def optimize_cutoff_time():
    """
    This example includes the optimization of the memory storage cut-off time.
    Without cut-off, this parameters give zero secret rate.
    With the optimized cut-off, the secret key rate can be increased to
    more than 3*10^(-3).
    Depending on the hardware, running the whole example may take a few hours.
    The uniform cut-off optimization is smaller.
    """
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.98,
        "t_coh": 400,
        "t_trunc": 3000,
        "cut_type": "memory_time",
        }
    log_init("opt", level=logging.INFO)

    # Uniform cut-off optimization, ~2.5 min on Intel i7 8700
    logging.info("Uniform cut-off optimization\n")
    # Define optimizer parameters
    opt = CutoffOptimizer(opt_kind="uniform_de", adaptive=True, tolerance=0.)
    # Run optimization
    best_tau = opt.run(parameters, tau_dims=1)
    # Calculate the secret key rate
    parameters["tau"] = best_tau
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.5f}".format(key_rate))

    # Nonuniform cut-off optimization, 20~30 min on Intel i7 8700
    logging.info("Nonuniform cut-off optimization\n")
    tau_dims = len(parameters["protocol"])
    opt = CutoffOptimizer(opt_kind="full_de", adaptive=True, tolerance=0.)
    best_tau = opt.run(parameters, tau_dims=tau_dims)
    parameters["tau"] = best_tau
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.5f}".format(key_rate))

    logging.info("No cut-off\n")
    parameters["tau"] = np.iinfo(np.int).max
    pmf, w_func = repeater_sim(parameters=parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate without cut-off: {:.5f}".format(key_rate))
    logging.info("Rate without truncation time: {}\n".format(key_rate))
single_parameter_simulation()