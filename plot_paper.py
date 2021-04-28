from copy import deepcopy
import logging
import time
import multiprocessing as mp
from functools import partial

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import axis3d
import matplotlib.gridspec as gridspec
import seaborn as sns


from optimize_cutoff import (optimization_tau_wrapper, parallel_tau_warpper,
    uniform_tau_pretrain, full_tau_pretrain_high_tau)
from utility_functions import secret_key_rate, werner_to_fid, get_mean_werner, get_mean_waiting_time
from logging_utilities import log_init, log_params, log_finish, printProgressBar, save_data, load_data, find_record_id, find_record_patterns
from repeater_algorithm import repeater_sim, plot_algorithm, RepeaterChainSimulation
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import *
from matplotlib import cm

TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487

#######################################################################
# fig 4
def plot_swap_with_cutoff_data():
    """
    Gathering data for the two cases. Run time ~ a few minutes.
    """
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.0001,
        "p_swap": 0.5,
        "mt_cut": [10000000000, (17000, 32000, 55000)],
        "w0": 0.98,
        "t_coh": 400000,
        "t_trunc": 3000000,
        "sample_size": 10000000,
        }

    kwarg_list = create_iter_kwargs(parameters)

    pmf_list = []
    w_func_list = []

    # MC
    for kwarg in kwarg_list:
        start = time.time()
        print("Sample parameters:")
        print(kwarg)
        pmf, w_func = repeater_mc(kwarg, return_pmf=True)
        end = time.time()
        print("MC elapse time\n", end-start)
        pmf_list.append(pmf)
        w_func_list.append(w_func)

    # exact
    for kwarg in kwarg_list:
        start = time.time()
        pmf, w_func = repeater_sim(parameters=kwarg)
        end = time.time()
        t = 0
        while(pmf[t]<1.0e-17):
            w_func[t] = np.nan
            t += 1
        print("Deterministic elapse time\n", end-start)
        print()
        pmf_list.append(pmf)
        w_func_list.append(w_func)
    np.save("figures/swap_with_cutoff", [pmf_list, w_func_list])


def plot_swap_with_cutoff_fig():
    sns.set_palette("Paired")

    pmf_list, w_func_list = np.load("figures/swap_with_cutoff.npy", allow_pickle=True)
    fig = plt.figure(figsize=(LINEWIDTH, LINEWIDTH*3.5/5), dpi=150)

    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.0, hspace=0.00)
    axis = (plt.subplot(gs[0]), plt.subplot(gs[1]))

    max_plot_t = 1400000
    plot_step = 1000
    prob_scale = 100000
    pmf = prob_scale*pmf_list[0,]
    average_pmf = np.array([np.sum(pmf[i*plot_step: i*plot_step+plot_step])/plot_step for i in range(0, int(max_plot_t/plot_step))])
    axis[0].plot(average_pmf, marker='.',markersize=2.5, linewidth=0)
    w_func = w_func_list[0,]
    average_w = []
    for i in range(0, int(max_plot_t/plot_step)):
        t_array = np.arange(i*plot_step, (i+1)*plot_step)
        temp = w_func[i*plot_step: i*plot_step+plot_step]
        temp = temp[~np.isnan(temp)]
        average_w.append(np.sum(temp)/len(temp))
    average_w = np.asarray(average_w)
    axis[1].plot(werner_to_fid(average_w), marker='.', markersize=2.5, linewidth=0)
    axis[0].plot(prob_scale*pmf_list[2,][: max_plot_t:plot_step], linewidth=0.7, label="without cut-off")
    w_func_list[2,][:1000] = np.nan  # numerical instability
    axis[1].plot(werner_to_fid(w_func_list[2,][: max_plot_t:plot_step]), linewidth=0.7)
    # shift the color
    l1 = axis[0].plot([0])
    l2 = axis[1].plot([1])
    l3 = axis[0].plot([0])
    l4 = axis[1].plot([1])
    l1 = axis[0].plot([0])
    l2 = axis[1].plot([1])
    l3 = axis[0].plot([0])
    l4 = axis[1].plot([1])

    # with cutoff
    # pmf + MC
    pmf = prob_scale*pmf_list[1,]
    average_pmf = np.array([np.sum(pmf[i*plot_step: i*plot_step+plot_step])/plot_step for i in range(0, int(max_plot_t/plot_step))])
    axis[0].plot(average_pmf, '.',markersize=2.5, linewidth=0)
    # werner + MC
    w_func = w_func_list[1,]
    average_w = []
    for i in range(0, int(max_plot_t/plot_step)):
        t_array = np.arange(i*plot_step, (i+1)*plot_step)
        temp = w_func[i*plot_step: i*plot_step+plot_step]
        temp = temp[~np.isnan(temp)]
        average_w.append(np.sum(temp)/len(temp))
    average_w = np.asarray(average_w)
    average_w[:3] = np.nan
    axis[1].plot(werner_to_fid(average_w), '.',markersize=2.5, linewidth=0)
    # pmf + algorithm
    axis[0].plot(prob_scale*pmf_list[3,][: max_plot_t:plot_step], linewidth=0.7, label="with cut-off")
    # werner + algorithm
    w_func_list[3,][:1000] = np.nan  # numerical instability
    axis[1].plot(werner_to_fid(w_func_list[3,][: max_plot_t:plot_step]), linewidth=0.7)
    # plot setup
    del l1, l2, l3, l4
    axis[0].set_ylabel(r"$\Pr(T=t)$"+" "+r"$(10^{-5})$")
    axis[1].set_ylabel(r"Fidelity $F(t)$")
    axis[1].set_xlabel(r"Waiting time t $(10^5)$")
    axis[1].set_xticklabels([0, 0, 2, 4, 6, 8, 10, 12, 14])
    axis[0].set_xticks([])
    axis[0].set_xticklabels([])
    axis[0].legend(fontsize="small")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig("figures/swap_with_cutoff.png")
    fig.savefig("figures/swap_with_cutoff.pdf")
    fig.show()
    return fig


###############################################################################
# fig 5
def plot_trade_off_data():
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.001,
        "p_swap": 0.5,
        "mt_cut": 10000000,
        "w0": 0.98,
        "t_coh": 40000,
        "t_trunc": 500000
        }

    t_trunc = parameters["t_trunc"]
    tau_list = np.array(np.arange(2500, 15000, 100))

    pmf_matrix, w_func_matrix = parallel_tau_warpper(tau_list, parameters, t_trunc, workers=10)

    np.save("figures/trade_off", [pmf_matrix, w_func_matrix])


def plot_trade_off_fig():
    pmf_matrix, w_func_matrix = np.load("figures/trade_off.npy")
    t_trunc = 500000
    tlist = np.arange(t_trunc)
    tau_list = np.array(np.arange(2500, 15000, 100))
    cdf_matrix = np.cumsum(pmf_matrix, axis=1)

    # compute rate, mean fidelity and secret key rate
    aver_w_list = []
    raw_rate_list = []
    secret_key_rate_list = []
    for i, tau in enumerate(tau_list):
        tlist = np.arange(t_trunc)
        aver_T = get_mean_waiting_time(pmf_matrix[i])
        raw_rate_list.append(1./aver_T)
        aver_w_list.append(get_mean_werner(pmf_matrix[i], w_func_matrix[i]))
        secret_key_rate_list.append(secret_key_rate(pmf_matrix[i], w_func_matrix[i]))
    aver_fid_list = werner_to_fid(np.array(aver_w_list))

    # plot
    fig = plt.figure(figsize=(LINEWIDTH,LINEWIDTH*1.1*0.618), dpi = 200)
    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.0, hspace=0.05)
    axis1, axis2 = (plt.subplot(gs[0]), plt.subplot(gs[1]))
    # plot time and fidelity
    a, = axis1.plot(tau_list, aver_fid_list, "--", color="slategrey", label=r"$\bar{F}$")
    ax2 = axis1.twinx()  # instantiate a second axis that shares the same x-axis
    b, = ax2.plot(tau_list, np.array(raw_rate_list)*100000, color="slategrey", label=r"$1/\bar{T}$")
    ax2.set_ylabel(r"$1/\bar{T}$ $(10^{-5})$")
    axis1.set_ylabel(r"$\bar{F}$")
    axis1.set_xticks([])
    axis1.set_xticklabels([])
    ax2.text(11500, 1.8, r"$\bar{F}$", color='k', fontsize="small")
    ax2.text(11500, 3.2, r"$1/\bar{T}$", color='k', fontsize="small")
    # plot secret key rate
    axis2.plot(tau_list, np.array(secret_key_rate_list)*100000, color="slategrey")
    axis2.set_xlabel(r"Cut-off $\tau$")
    axis2.set_ylabel(r"R $(10^{-5})$")
    plt.subplots_adjust(bottom=0.14, left=0.14, top=0.95, right=0.87)
    fig.savefig("figures/trade_off.pdf")
    fig.savefig("figures/trade_off.png")
    fig.show()
    return fig


###############################################################################
# Collect data for fig 6 or 7
def parameter_regime_step(parameters, track, workers=1):
    parameters = deepcopy(parameters)
    simulator = RepeaterChainSimulation()
    simulator.use_gpu = True
    if parameters["optimizer"] == "uniform_de":
        opt = CutoffOptimizer(opt_kind="uniform_de", disp=True, adaptive=True, tol=0.01, workers=workers, simulator=simulator, sample_distance=parameters["sample_distance"])
        best_tau = opt.run(parameters)
    elif parameters["optimizer"] == "nonuniform_de":
        opt = CutoffOptimizer(opt_kind="nonuniform_de", disp=True, adaptive=True, tol=0.01, workers=workers, simulator=simulator, sample_distance=parameters["sample_distance"])
        best_tau = opt.run(parameters)
    elif parameters["optimizer"] == "none": # no cutoff
        best_tau = {"memory_time": np.iinfo(np.int32).max}
    else:
        raise ValueError("Unknown optimizer {}.".format(parameters["optimizer"]))

    return {"tau": best_tau}


def _parallel_warpper(parameters, data_dict):
    tau = data_dict[(parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])]["tau"]

    parameters["cutoff_dict"] = tau
    pmf, w_func = repeater_sim(parameters)
    return pmf, w_func


def complete_data(ID, parameters_list=None, workers=mp.cpu_count()-2):
    if parameters_list is None:
        parameters = find_record_id(ID)
        kwarg_list = create_iter_kwargs(parameters)
    else:
        kwarg_list = parameters_list
    data_dict = load_data(ID)

    pool = mp.Pool(workers)
    result = pool.map(partial(_parallel_warpper, data_dict=deepcopy(data_dict)), kwarg_list)
    pool.close()
    pool.join()

    for kwarg, (pmf, w_func) in zip(kwarg_list, result):
        temp = {}
        temp["pmf"] = pmf
        temp["w_func"] = w_func
        temp["key_rate"] = secret_key_rate(pmf, w_func)
        data_dict[(kwarg["p_gen"], kwarg["p_swap"], kwarg["w0"], kwarg["t_coh"], kwarg["optimizer"])].update(temp)
    outfile = open("data/" + ID + ".pickle", 'wb')
    pickle.dump(data_dict, outfile)
    outfile.close()


def parameter_regime(parameters_list, ID, workers=mp.cpu_count()-2, remark=""):
    """
    Optimize cutoffs over the list of given parameters and save the result
    in the data folder.
    If any keyword in `parameter` is a list, the list will be unfolded and
    the algorithm iterate over this list, with all other parameters fixed.
    The final result will be saved.
    There are five valid keywords considered:
    `p_gen`, `p_swap`, `w0`, `t_coh` and `optimizer`.
    """
    # Unfold the list
    if isinstance(parameters_list, dict):
        kwarg_list = create_iter_kwargs(parameters_list)
    else:
        kwarg_list = deepcopy(parameters_list)

    # run simulation
    data_dict = {}
    for i, parameters in enumerate(kwarg_list):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])
        data_dict[key] = parameter_regime_step(parameters, False, workers=workers)
        save_data(ID, data_dict)

    # finish
    variable = {}
    for key, value in parameters_list.items():
        if isinstance(value, list):
            variable[key] = value
    parameters_list["variable"] = variable
    complete_data(ID, parameters_list=kwarg_list, workers=workers)
    log_finish(ID, parameters_list, remark)

###############################################################################
# fig 6
def get_zero_keyrate_borderline(remark=None):
    remark="fourier_parameter_regime"
    if remark is not None:
        parameters_list = find_record_patterns({"remark": remark})
    if not isinstance(parameters_list, list):
        parameters_list = [parameters_list]

    w0_array = np.array([], dtype=np.float)
    t_coh_array = np.array([], dtype=np.int)
    data = {}
    for parameters in parameters_list:
        ID = parameters["ID"]
        data.update(load_data(ID))
        w0_array = np.concatenate([w0_array, parameters["w0"]])
        t_coh_array = np.concatenate([t_coh_array, parameters["t_coh"]])
    w0_array = np.unique(np.sort(w0_array))
    t_coh_array = np.unique(np.sort(t_coh_array))

    result = []
    for w0 in w0_array:
        highest_key_rate = 0.
        best_t_coh = 0
        for t_coh in t_coh_array:
            key1 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "nonuniform_de")
            key2 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "none")
            key_rate = data[key1]["key_rate"] - data[key2]["key_rate"]
            if key_rate > highest_key_rate:
                best_t_coh = t_coh
                highest_key_rate = key_rate
            else:
                break
        
        t_coh = best_t_coh
        key_rate = 0.
        while key_rate == 0.:
            t_coh += 50
            parameters["w0"] = w0
            parameters["t_coh"] = t_coh
            pmf, w_func = repeater_sim(parameters)
            key_rate = secret_key_rate(pmf, w_func)
        result.append((w0, t_coh))
    np.save("figures/zero_keyrate_borderline.npy",
        [[result[i][0] for i in range(len(result))],
        [result[i][1] for i in range(len(result))]]
        )

def plot_parameter_contour(remark):
    sns.set_palette("Blues")
    if remark is not None:
        parameters_list = find_record_patterns({"remark": remark})
    if not isinstance(parameters_list, list):
        parameters_list = [parameters_list]

    w0_array = np.array([], dtype=np.float)
    t_coh_array = np.array([], dtype=np.int)
    data = {}
    for parameters in parameters_list:
        ID = parameters["ID"]
        data.update(load_data(ID))
        w0_array = np.concatenate([w0_array, parameters["w0"]])
        t_coh_array = np.concatenate([t_coh_array, parameters["t_coh"]])
    w0_array = np.unique(np.sort(w0_array))
    t_coh_array = np.unique(np.sort(t_coh_array))

    t_coh_mesh, w0_mesh = np.meshgrid(t_coh_array, w0_array)
    num_w0 = len(w0_array)
    num_t_coh = len(t_coh_array)

    key_rate_list = []
    for t_coh, w0 in zip(t_coh_mesh.reshape(num_t_coh*num_w0), w0_mesh.reshape(num_t_coh*num_w0)):
        key1 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "nonuniform_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "none")
        key_rate = data[key1]["key_rate"] - data[key2]["key_rate"]
        key_rate_list.append(key_rate)
    key_rate_mesh = np.asarray(key_rate_list).reshape(num_w0, num_t_coh)

    fig, axis = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH*0.8), dpi=200)
    cs = axis.contourf(t_coh_array, werner_to_fid(np.array(w0_array)), key_rate_mesh, cmap="Blues", levels=9)
    cbar = fig.colorbar(cs)
    edge_w0, edge_t_coh = np.load("figures/zero_keyrate_borderline.npy")
    axis.plot(edge_t_coh, werner_to_fid(edge_w0), 'k')
    axis.set_xlabel(r"Coherence time $t_{\rm{coh}}$")
    axis.set_ylabel(r"Initial Fidelity")
    cbar.ax.set_ylabel(r"Increase in the secret key rate")
    fig.tight_layout()
    fig.savefig("figures/parameter_regime.pdf")
    fig.savefig("figures/parameter_regime.png")
    fig.show()


###############################################################################
# fig 7
def parameter_regime_slice_fig1(ID):
    sns.set_palette("Dark2")

    parameters = find_record_id(ID)
    data = load_data(ID)

    w0_array = parameters["w0_array"]
    t_coh_array = parameters["t_coh_array"]
    t_coh_mesh, w0_mesh = np.meshgrid(t_coh_array, w0_array)
    size_2d = (len(t_coh_array), len(w0_array))
    size_1d = len(t_coh_array) * len(w0_array)

    lyl_best_key_list = np.empty(len(t_coh_array))
    full_best_key_list = np.empty(len(t_coh_array))
    unique_best_key_list = np.empty(len(t_coh_array))
    no_tau_key_list = np.empty(len(t_coh_array))
    w0 = 0.98
    for i, t_coh in enumerate(t_coh_array):
        single_round_data  = data[(t_coh, w0)]
        lyl_best_tau,lyl_best_key = get_tau_and_key_rate(single_round_data, "lbl_tau_opt_data")
        unique_best_tau, unique_best_key = get_tau_and_key_rate(single_round_data, "unique_tau_opt_data")
        full_best_tau, full_best_key = get_tau_and_key_rate(single_round_data, "full_tau_opt_data")
        no_best_tau, no_best_key = get_tau_and_key_rate(single_round_data, "no_tau_data")
        lyl_best_key_list[i] = lyl_best_key
        full_best_key_list[i] = full_best_key
        unique_best_key_list[i] = unique_best_key
        no_tau_key_list[i] = no_best_key

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(LINEWIDTH, LINEWIDTH*0.618))
    
    ax1.plot(t_coh_array, no_tau_key_list*1000, ".-", label="without cut-off")
    # ax1.plot(t_coh_array, unique_best_key_list, ".-", label="same cut-off")
    ax1.plot(t_coh_array, lyl_best_key_list*1000, ".-", label="lyl cut-off")
    ax1.plot(t_coh_array, full_best_key_list*1000, ".-", label="with cut-off")
    ax1.set_xlabel(r"$t_{\rm{coh}}$")
    ax1.set_ylabel(r"Secret key rate ($10^{-3}$)")
    ax1.set_ylim(ax1.get_ylim()[0], 1.25)
    # ax1.legend()
    fig.tight_layout()
    fig1 = fig


    parameters = find_record_id(ID)
    data = load_data(ID)

    w0_array = parameters["w0_array"]
    t_coh_array = parameters["t_coh_array"]
    t_coh_mesh, w0_mesh = np.meshgrid(t_coh_array, w0_array)
    size_2d = (len(t_coh_array), len(w0_array))
    size_1d = len(t_coh_array) * len(w0_array)

    lyl_best_key_list = np.empty(len(w0_array))
    full_best_key_list = np.empty(len(w0_array))
    unique_best_key_list = np.empty(len(w0_array))
    no_tau_key_list = np.empty(len(w0_array))
    t_coh = 400
    for i, w0 in enumerate(w0_array):
        single_round_data  = data[(t_coh, w0)]
        lyl_best_tau,lyl_best_key = get_tau_and_key_rate(single_round_data, "lbl_tau_opt_data")
        unique_best_tau, unique_best_key = get_tau_and_key_rate(single_round_data, "unique_tau_opt_data")
        full_best_tau, full_best_key = get_tau_and_key_rate(single_round_data, "full_tau_opt_data")
        no_best_tau, no_best_key = get_tau_and_key_rate(single_round_data, "no_tau_data")
        lyl_best_key_list[i] = lyl_best_key
        full_best_key_list[i] = full_best_key
        unique_best_key_list[i] = unique_best_key
        no_tau_key_list[i] = no_best_key

    ax2.plot(w0_array, no_tau_key_list*1000, ".-", label="without cut-off")
    # ax2.plot(t_coh_array, unique_best_key_list, ".-", label="same cut-off")
    ax2.plot(w0_array, lyl_best_key_list*1000, ".-", label="lyl cut-off")
    ax2.plot(w0_array, full_best_key_list*1000, ".-", label="with cut-off")
    ax2.set_xlabel(r"$w_0$")
    # ax2.set_ylabel(r"Secret key rate ($10^{-3}$)")
    ax2.set_ylim(ax2.get_ylim()[0], 1.25)
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    # ax2.legend()
    fig.tight_layout(w_pad=0.01)
    fig.savefig("figures/figures/parameter_regime_slice.pdf")
    fig.savefig("figures/parameter_regime_slice.png")
    fig2 = fig
    return fig1, fig2

def plot_sensitivity_parameters():
    default_parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.002,
        "p_swap": 0.5,
        "w0": 0.97,
        "t_coh": 35000,
        "t_trunc": 900000,
        "optimizer": ["nonuniform_de", "uniform_de", "none"],
        "sample_distance": 50
        }
    sns.set_palette("Dark2")
    fig, axs = plt.subplots(2, 4, figsize=(TEXTWIDTH, TEXTWIDTH/2), dpi=200)

    # p_gen
    keyword = {"remark": "fourier_sensitivity_p_gen"}
    parameters = find_record_patterns(keyword)
    print(parameters)
    ID = parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_gen"]))
    improvement_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "nonuniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key2 = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "default_nonuniform_de")
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
    axs[1][0].plot(parameters["p_gen"], -improvement_list, '*', label="non-uniform")
    axs[0][0].plot(parameters["p_gen"], key_rate_list * 1.0e5, '*', label="non-uniform")

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_gen"]))
    improvement_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key2 = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "default_uniform_de")
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
    axs[1][0].plot(parameters["p_gen"], -improvement_list, '+', label="uniform")
    axs[0][0].plot(parameters["p_gen"], key_rate_list * 1.0e5, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][0].plot(parameters["p_gen"], no_timeout_key_rate_list * 1.0e5, '.', label="no tau")

    axs[0][0].set_xticklabels([])
    axs[0][0].set_ylabel(r"$R(\tau_{\rm{target}}) \quad (10^{-5})$")
    axs[0][0].text(0.005, 0., "(a)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][0].text(0.005, 0.9, "(e)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][0].plot(default_parameters["p_gen"], 0, '.', label="No cut-off")
    axs[1][0].plot(default_parameters["p_gen"], 0, 'o')
    axs[1][0].set_ylim((-0.05, 1.05))
    axs[1][0].set_ylabel(r"Relative $R$ improvement of"+"\n"+r"$\tau_{\rm{target}}$ vs. $\tau_{\rm{baseline}}$")
    axs[1][0].legend(fontsize="x-small", loc = 2)
    axs[1][0].set_xlabel(r"$p_{\rm{gen}}$")
    # axs[0][0].legend()

    # p_swap
    keyword = {"remark": "fourier_sensitivity_p_swap"}
    parameters = find_record_patterns(keyword)
    ID = parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_swap"]))
    improvement_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "nonuniform_de")
        key2 = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "default_nonuniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
    key_rate_list = np.array(key_rate_list)
    axs[1][1].plot(parameters["p_swap"], -improvement_list, '*', label="non-uniform")
    axs[0][1].plot(parameters["p_swap"], key_rate_list * 1.0e5, '*', label="non-uniform")

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_swap"]))
    improvement_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "uniform_de")
        key2 = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "default_uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
    key_rate_list = np.array(key_rate_list)
    axs[1][1].plot(parameters["p_swap"], -improvement_list, '+', label="uniform")
    axs[0][1].plot(parameters["p_swap"], key_rate_list * 1.0e5, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "none")
        # tau = data_dict[key]["tau"]
        # pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][1].plot(parameters["p_swap"], no_timeout_key_rate_list * 1.0e5, '.', label="no tau")

    axs[1][1].plot(default_parameters["p_swap"], 0, '.')
    axs[1][1].plot(default_parameters["p_swap"], 0, 'o')
    axs[1][1].set_ylim((-0.05, 1.05))
    # axs[1][1].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][1].legend()
    axs[0][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel(r"$p_{\rm{swap}}$")
    # axs[0][1].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][1].legend()
    axs[0][1].text(0.8, 0., "(b)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][1].text(0.8, 0.9, "(f)", horizontalalignment='right', verticalalignment='bottom')

    # w0
    keyword = {"remark": "fourier_sensitivity_w0"}
    parameters = find_record_patterns(keyword)
    ID= parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["w0"]))
    improvement_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "nonuniform_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "default_nonuniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
        improvement_list[i] = max(improvement_list[i], -1.)
    axs[1][2].plot(parameters["w0"], -improvement_list, '*', label="non-uniform")
    axs[0][2].plot(parameters["w0"], key_rate_list * 1.0e5, '*', label="non-uniform")
    
    tau_list = []
    key_rate_list = np.empty(len(parameters["w0"]))
    improvement_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "uniform_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "default_uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
        improvement_list[i] = max(improvement_list[i], -1.)
    axs[1][2].plot(parameters["w0"], -improvement_list, '+', label="uniform")
    axs[0][2].plot(parameters["w0"], key_rate_list * 1.0e5, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "none")
        # tau = data_dict[key]["tau"]
        # pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][2].plot(parameters["w0"], no_timeout_key_rate_list * 1.0e5, '.', label="no tau")

    axs[1][2].plot(default_parameters["w0"], 0, '.')
    axs[1][2].plot(default_parameters["w0"], 0, 'o')
    axs[1][2].set_ylim((-0.05, 1.05))
    # axs[1][2].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][2].legend()
    axs[0][2].set_xticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][2].set_xlabel(r"$w_{\rm{0}}$")
    # axs[0][2].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][2].legend()
    axs[0][2].text(0.99, 0., "(c)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][2].text(0.99, 0.9, "(g)", horizontalalignment='right', verticalalignment='bottom')

    # t_coh
    keyword = {"remark": "fourier_sensitivity_t_coh"}
    parameters = find_record_patterns(keyword)
    ID= parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["t_coh"]))
    improvement_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "nonuniform_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "default_nonuniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
        improvement_list[i] = max(improvement_list[i], -1.)
    axs[1][3].plot(parameters["t_coh"], -improvement_list, '*', label="non-uniform")
    axs[0][3].plot(parameters["t_coh"], key_rate_list * 1.0e5, '*', label="non-uniform")

    tau_list = []
    key_rate_list = np.empty(len(parameters["t_coh"]))
    improvement_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "uniform_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "default_uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        key_rate_list[i] = data_dict[key]["key_rate"]
        key_rate_with_default_cutoff = data_dict[key2]["key_rate"]
        improvement_list[i] = (key_rate_with_default_cutoff-key_rate_list[i])/key_rate_with_default_cutoff
        improvement_list[i] = max(improvement_list[i], -1.)

    axs[1][3].plot(parameters["t_coh"], -improvement_list, '+', label="uniform")
    axs[0][3].plot(parameters["t_coh"], key_rate_list * 1.0e5, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][3].plot(parameters["t_coh"], no_timeout_key_rate_list * 1.0e5, '.', label="no tau")

    axs[1][3].plot(default_parameters["t_coh"], 0, '.')
    axs[1][3].plot(default_parameters["t_coh"], 0, 'o')
    axs[1][3].set_ylim((-0.05, 1.05))
    # axs[1][3].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][3].legend()
    axs[0][3].set_xticklabels([])
    axs[0][3].set_yticks([0, 0.5, 1. ,1.5])
    axs[1][3].set_yticklabels([])
    axs[1][3].set_xlabel(r"$t_{\rm{coh}}$")
    # axs[0][3].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][3].legend()
    axs[0][3].text(100000, 0., "(d)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][3].text(100000, 0.9, "(h)", horizontalalignment='right', verticalalignment='bottom')

    plt.subplots_adjust(top=0.98, bottom=0.1, right=0.99, left=0.08)
    fig.savefig("figures/tau_sensitivity.pdf")
    fig.savefig("figures/tau_sensitivity.png")
    return fig

def calculate_key_rate_with_default_cutoff(ID, parameters, temp_parameters):
    data_dict = load_data(ID)
    temp_parameters = deepcopy(temp_parameters)
    optimizers = ["nonuniform_de", "uniform_de"]
    temp_parameters.pop("optimizer")
    for optimizer in optimizers:
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], optimizer)
        temp_parameters["cutoff"] = data_dict[key]["tau"]["memory_time"]
        for kwargs in create_iter_kwargs(temp_parameters):
            pmf, w_func = repeater_sim(kwargs)
            key = (kwargs["p_gen"], kwargs["p_swap"], kwargs["w0"], kwargs["t_coh"], "default_"+optimizer)
            temp = {}
            temp["key_rate"] = secret_key_rate(pmf, w_func)
            data_dict[key] = temp

    current_level = logging.getLogger().level
    logging.getLogger().level = logging.EXP
    save_data(ID, data_dict)
    logging.getLogger().level = current_level

if __name__ == "__main__":

    # plt.style.use("classic")
    plt.rcParams.update(
        {"font.size": 9,
        # "font.family": "Arial",
        'legend.fontsize': 'x-small',
        'axes.labelsize': 'small',
        # 'axes.titlesize':'x-small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small',
        })

    # # fig 4
    # plot_swap_with_cutoff_data()
    plot_swap_with_cutoff_fig()

    # # fig 5
    # # plot_trade_off_data()
    # plot_trade_off_fig()

    # # fig8
    # parameters = {
    #     "protocol": (0, 0, 0),
    #     "p_gen": 0.001,
    #     "p_swap": 0.5,
    #     "w0": [0.97, 0.975,0.98, 0.985, 0.99, 0.995, 1.0],
    #     "t_coh": [22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 42500,45000, 47500, 50000, 55000, 60000, 70000, 80000, 100000, 130000],
    #     "t_trunc": 900000,
    #     "optimizer": ["nonuniform_de", "none"],
    #     "sample_distance": 50
    #     }

    # # ID = log_init("optimize", level=logging.EXP)
    # # log_params(parameters)
    # # parameter_regime(parameters, ID, workers=8, remark="fourier_parameter_regime")

    # # get_zero_keyrate_borderline("fourier_parameter_regime")

    # plot_parameter_contour("fourier_parameter_regime")

    # # fig 7
    # logging_level = logging.EXP
    # parameters = {
    #     "protocol": (0, 0, 0),
    #     "p_gen": 0.002,
    #     "p_swap": 0.5,
    #     "w0": 0.97,
    #     "t_coh": 35000,
    #     "t_trunc": 900000,
    #     "optimizer": ["nonuniform_de", "uniform_de", "none"],
    #     "sample_distance": 50
    #     }

    # ID = log_init("optimize", level=logging_level)
    # temp_parameters = deepcopy(parameters)
    # log_params(temp_parameters)
    # temp_parameters["t_coh"] = list(np.trunc(np.linspace(15000, 100000, 18)).astype(np.int))
    # parameter_regime(temp_parameters, ID, workers=8, remark="fourier_sensitivity_t_coh")
    # calculate_key_rate_with_default_cutoff(ID, parameters, temp_parameters)

    # ID = log_init("optimize", level=logging_level)
    # temp_parameters = deepcopy(parameters)
    # temp_parameters["p_gen"] = list(np.linspace(0.0005, 0.005, 10))
    # parameter_regime(temp_parameters, ID, workers=8, remark="fourier_sensitivity_p_gen")
    # calculate_key_rate_with_default_cutoff(ID, parameters, temp_parameters)

    # ID = log_init("optimize", level=logging_level)
    # temp_parameters = deepcopy(parameters)
    # log_params(temp_parameters)
    # temp_parameters["p_swap"] = list(np.linspace(0.3, 0.8, 11))
    # parameter_regime(temp_parameters, ID, workers=8, remark="fourier_sensitivity_p_swap")
    # calculate_key_rate_with_default_cutoff(ID, parameters, temp_parameters)

    # ID = log_init("optimize", level=logging_level)
    # temp_parameters = deepcopy(parameters)
    # log_params(temp_parameters)
    # temp_parameters["w0"] = list(np.linspace(0.96, 0.99, 13))
    # parameter_regime(temp_parameters, ID, workers=8, remark="fourier_sensitivity_w0")
    # calculate_key_rate_with_default_cutoff(ID, parameters, temp_parameters)

    # fig = plot_sensitivity_parameters()
    # fig.show()



