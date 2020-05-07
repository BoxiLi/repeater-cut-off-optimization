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


from optimize_cutoff import (optimization_tau_wrapper, parallel_tau_warpper, full_tau_pretrain_high_tau)
from utility_functions import secret_key_rate, werner_to_fid
from logging_utilities import log_init, log_params, log_finish, printProgressBar, save_data, load_data, find_record_id, find_record_patterns
from repeater_algorithm import repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import *
from matplotlib import cm

TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487

#######################################################################

def plot_swap_with_cutoff_data():
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.001,
        "p_swap": 0.5,
        "tau": [10000000000, (1700, 3200, 5500)],
        "w0": 0.98,
        "t_coh": 40000,
        "disc_kind": "both",
        "t_trunc": 300000,
        "sample_size": 1000000000,
        "reuse_sampled_data": False
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
    [pmf_list_, w_func_list_] = np.load("swap_with_cutoff.npy", allow_pickle=True)
    pmf_list += list(pmf_list_)[2:4]
    w_func_list += list(w_func_list_)[2:4]
    np.save("figures\\swap_with_cutoff", [pmf_list, w_func_list])


def plot_swap_with_cutoff_fig():
    sns.set_palette("Paired")

    pmf_list, w_func_list = np.load("figures\\swap_with_cutoff.npy", allow_pickle=True)
    fig = plt.figure(figsize=(LINEWIDTH, LINEWIDTH*4/5), dpi=150)

    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.0, hspace=0.00)
    axis = (plt.subplot(gs[0]), plt.subplot(gs[1]))

    max_plot_t = 140000
    plot_step = 10
    prob_scale = 10000
    axis[0].plot(prob_scale*pmf_list[0,][: max_plot_t:plot_step], marker='.',markersize=2.5, linewidth=0)
    axis[1].plot(werner_to_fid(w_func_list[0,][: max_plot_t:plot_step]), marker='.', markersize=2.5, linewidth=0)
    axis[0].plot(prob_scale*pmf_list[2,][: max_plot_t:plot_step], linewidth=0.7, label="without cut-off")
    axis[1].plot(werner_to_fid(w_func_list[2,][: max_plot_t:plot_step]), linewidth=0.7)
    l1 = axis[0].plot([0])
    l2 = axis[1].plot([1])
    l3 = axis[0].plot([0])
    l4 = axis[1].plot([1])
    l1 = axis[0].plot([0])
    l2 = axis[1].plot([1])
    l3 = axis[0].plot([0])
    l4 = axis[1].plot([1])
    axis[0].plot(prob_scale*pmf_list[1,][: max_plot_t:plot_step], '.',markersize=2.5, linewidth=0)

    axis[1].plot(werner_to_fid(w_func_list[1,][: max_plot_t:plot_step]), '.',markersize=2.5, linewidth=0)
    axis[0].plot(prob_scale*pmf_list[3,][: max_plot_t:plot_step], linewidth=0.7, label="with cut-off")
    axis[1].plot(werner_to_fid(w_func_list[3,][: max_plot_t:plot_step]), linewidth=0.7)
    print(secret_key_rate(pmf_list[3, ], w_func_list[3, ]))
    # plot setup
    del l1, l2, l3, l4
    axis[0].set_ylabel(r"$\Pr(T=t)$"+" "+r"$(10^{-4})$")
    axis[1].set_ylabel(r"Fidelity $F(t)$")
    axis[1].set_xlabel(r"Waiting time t $(10^4)$")
    axis[1].set_xticklabels([0, 0, 2, 4, 6, 8, 10, 12, 14])
    axis[0].set_xticks([])
    axis[0].set_xticklabels([])
    axis[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig("figures\\swap_with_cutoff.png")
    fig.savefig("figures\\swap_with_cutoff.pdf")
    return fig
###############################################################################
# fig 5
# plot_trade_off_data()
# plot_trade_off_fig()
def plot_trade_off_data():
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "tau": 100000,
        "sample_size": 1000,
        "w0": 0.98,
        "t_coh": 400,
        "disc_kind": "both",
        "reuse_sampled_data": False,
        "t_trunc": 4000
        }

    t_trunc = parameters["t_trunc"]
    tau_list = np.array(np.arange(25, 150, 1))

    pmf_matrix, w_func_matrix = parallel_tau_warpper(tau_list, parameters, t_trunc)

    np.save("figures\\trade_off", [pmf_matrix, w_func_matrix])


def plot_trade_off_fig():
    pmf_matrix, w_func_matrix = np.load("figures\\trade_off.npy")
    t_trunc = 4000
    tlist = np.arange(t_trunc)
    tau_list = np.array(np.arange(30, 150, 1))
    cdf_matrix = np.cumsum(pmf_matrix, axis=1)
    valid_tau_list = []
    aver_w = []
    raw_rate = []
    secret_key_rate_list = []
    for i, tau in enumerate(tau_list):
        thresh_ind = np.searchsorted(cdf_matrix[i], 0.99)
        if thresh_ind != len(tlist):
            valid_tau_list.append(tau)
            aver_w.append(np.sum(w_func_matrix[i, 1:thresh_ind] * pmf_matrix[i, 1:thresh_ind]))
            aver_t = 1./np.sum(np.arange(t_trunc) * pmf_matrix[i])
            raw_rate.append(aver_t)
            r = secret_key_rate(pmf_matrix[i], w_func_matrix[i])
            secret_key_rate_list.append(r)
    fig = plt.figure(figsize=(LINEWIDTH,LINEWIDTH*1.1*0.618), dpi = 200)

    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.0, hspace=0.00)
    axis1, axis2 = (plt.subplot(gs[0]), plt.subplot(gs[1]))
    # (2, 1, figsize=(LINEWIDTH,LINEWIDTH*1.5*0.618), dpi = 200)
    a, = axis1.plot(valid_tau_list, werner_to_fid(np.array(aver_w)), "--", color="slategrey", label=r"$\bar{F}$")
    ax2 = axis1.twinx()  # instantiate a second axis that shares the same x-axis
    b, = ax2.plot(valid_tau_list, np.array(raw_rate)*1000, color="slategrey", label=r"$1/\bar{T}$")
    ax2.set_ylabel(r"$1/\bar{T}$ $(10^{-3})$")
    axis1.set_ylabel(r"$\bar{F}$")
    axis1.set_xticks([])
    axis1.set_xticklabels([])
    ax2.text(100, 2.0, r"$\bar{F}$", color='k', fontsize="small")
    ax2.text(115, 3.3, r"$1/\bar{T}$", color='k', fontsize="small")

    valid_tau_list = []
    secret_key_rate_list = []
    for i, tau in enumerate(tau_list):
        thresh_ind = np.searchsorted(cdf_matrix[i], 0.99)
        if thresh_ind != len(tlist):
            valid_tau_list.append(tau)
            r = secret_key_rate(pmf_matrix[i], w_func_matrix[i])
            secret_key_rate_list.append(r)

    axis2.plot(valid_tau_list, np.array(secret_key_rate_list)*1000, color="slategrey")
    axis2.set_xlabel(r"Cut-off $\tau$")
    axis2.set_ylabel(r"R $(10^{-3})$")
    plt.subplots_adjust(bottom=0.14, left=0.14, top=0.95, right=0.87)
    fig.savefig("figures\\trade_off.pdf")
    fig.savefig("figures\\trade_off.png")
    fig.show()
    return fig

###############################################################################
# Collect data for fig 6 or 7

def parameter_regime_step(parameters, track, workers=1):
    current_log_level = logging.getLogger().level
    if track:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    parameters = deepcopy(parameters)
    # best_tau = optimization_by_level(parameters, disp=False)
    # lbl_tau_opt_data = {"parameters": parameters, "pmf": best_pmf, "w_func": best_w_func, "tau": best_tau, "key_rate": best_secret_key_rate, "remark":"Different tau for each level"}
    if parameters["optimizer"] == "uniform_de":
        opt = CutoffOptimizer(opt_kind="uniform_de", disp=False, adaptive=True, tolerance=1.0e-4, use_tracker=True, workers=workers)
        best_tau = opt.run(parameters, tau_dims=1)
    elif parameters["optimizer"] == "full_de":
        opt = CutoffOptimizer(opt_kind="full_de", disp=False, adaptive=True,  tolerance=1.0e-4, use_tracker=True, workers=workers)
        best_tau = opt.run(parameters, tau_dims=None)
    elif parameters["optimizer"] == "none":
        best_tau = np.iinfo(np.int32).max
    else:
        raise ValueError("Unknown optimizer {}.".format(parameters["optimizer"]))
    # data_dict[(parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])] = {"tau": best_tau}

    logging.getLogger().setLevel(current_log_level)
    return {"tau": best_tau}


def _parallel_warpper(parameters, data_dict):
    tau = data_dict[(parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])]["tau"]
    if parameters["optimizer"] == "full_de":
        # pmf, w_func = optimization_tau_wrapper(tau, repeater_sim, parameters)
        parameters["tau"] = tau
        pmf, w_func = repeater_sim(parameters)
        # print(parameters)
    elif parameters["optimizer"] == "uniform_de":
        pmf, w_func = optimization_tau_wrapper(tau, repeater_sim, parameters)
    elif parameters["optimizer"] == "none":
        pmf, w_func = optimization_tau_wrapper(tau, repeater_sim, parameters)
    else:
        raise ValueError("Unknown optimizer")

    return pmf, w_func


def complete_data(ID):
    parameters = find_record_id(ID)
    kwarg_list = create_iter_kwargs(parameters)
    data_dict = load_data(ID)

    pool = mp.Pool(mp.cpu_count()-1)
    result = pool.map(partial(_parallel_warpper, data_dict=deepcopy(data_dict)), kwarg_list)
    pool.close()
    pool.join()

    for kwarg, (pmf, w_func) in zip(kwarg_list, result):
        temp = {}
        temp["pmf"] = pmf
        temp["w_func"] = w_func
        temp["key_rate"] = secret_key_rate(pmf, w_func)
        data_dict[(kwarg["p_gen"], kwarg["p_swap"], kwarg["w0"], kwarg["t_coh"], kwarg["optimizer"])].update(temp)
    parameters = find_record_id(ID)
    outfile = open("data/" + ID + ".pickle", 'wb')
    pickle.dump(data_dict, outfile)
    outfile.close()


def parameter_regime(parameters_list, track=False, trunc_map=None, remark=""):
    ID = log_init("tau_opt", level=logging.EXP)

    log_params(parameters_list)
    kwarg_list = create_iter_kwargs(parameters_list)

    # manager = mp.Manager()
    # queue = manager.Queue()
    if track:
        pool = mp.Pool(mp.cpu_count()-2)

        jobs = []
        for i, parameters in enumerate(kwarg_list):
            if trunc_map is not None:
                parameters["t_trunc"] = trunc_map[1].get(parameters[trunc_map[0]], parameters["t_trunc"])
            # logging.info("step {}\n {}".format(i, kwarg))
            job = pool.apply_async(parameter_regime_step, (parameters, track))
            jobs.append(job)
            # printProgressBar(i, len(kwarg_list))
            # save_data(ID, data_dict)

        data_dict = {}
        for job, parameters in zip(jobs, kwarg_list):
            key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])
            data_dict[key] = job.get()
            save_data(ID, data_dict)

        #now we are done, kill the listener
        pool.close()
        pool.join()
    else:
        data_dict = {}
        for i, parameters in enumerate(kwarg_list):
            if trunc_map is not None:
                parameters["t_trunc"] = trunc_map[1].get(parameters[trunc_map[0]], parameters["t_trunc"])
            key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], parameters["t_coh"], parameters["optimizer"])
            if parameters["t_trunc"] > 4000:
                workers = 20
            else:
                workers = mp.cpu_count()-2
            data_dict[key] = parameter_regime_step(parameters, False, workers = mp.cpu_count()-2)
            save_data(ID, data_dict)

    # finishing
    variable = {}
    for key, value in parameters_list.items():
        if isinstance(value, list):
            variable[key] = value
    parameters_list["variable"] = variable
    log_finish(ID, parameters_list, remark=remark)
    complete_data(ID)
    return ID


###############################################################################
# fig 7
def plot_parameter_contour(ID=None):
    sns.set_palette("Blues")
    parameters = find_record_patterns({"remark": "exp parameter regime l=3"})
    ID = parameters["ID"]
    data1 = load_data(ID)

    parameters = find_record_patterns({"remark": "exp parameter regime l=3 none"})
    ID = parameters["ID"]
    data2 = load_data(ID)

    w0_array = parameters["w0"]
    t_coh_array = parameters["t_coh"]
    t_coh_mesh, w0_mesh = np.meshgrid(t_coh_array, w0_array)
    num_w0 = len(w0_array)
    num_t_coh = len(t_coh_array)

    key_rate_list = []
    for t_coh, w0 in zip(t_coh_mesh.reshape(num_t_coh*num_w0), w0_mesh.reshape(num_t_coh*num_w0)):
        key1 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "full_de")
        key2 = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, "none")

        key_rate = data1[key1]["key_rate"] - data2[key2]["key_rate"]
        key_rate_list.append(key_rate)
    key_rate_mesh = np.asarray(key_rate_list).reshape(num_w0, num_t_coh)

    fig, axis = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH*0.8), dpi=200)
    cs = axis.contourf(t_coh_array, werner_to_fid(np.array(w0_array)), key_rate_mesh, cmap="Blues")
    cbar = fig.colorbar(cs)
    edge_t_coh = np.array([586, 505, 443, 394, 355, 323, 295])
    edge_f0 = werner_to_fid(np.array([0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]))
    axis.plot(edge_t_coh, edge_f0, 'k')
    axis.set_xlim((300, 1500))
    axis.set_xlabel(r"Coherence time $t_{\rm{coh}}$")
    axis.set_ylabel(r"Initial Fidelity")
    cbar.ax.set_ylabel(r"Increase in the secret key rate $(10^{-2})$") 
    fig.tight_layout()
    fig.savefig("figures\\parameter_regime.pdf")
    fig.savefig("figures\\parameter_regime.png")
    fig.show()


###############################################################################
# fig 6

def plot_sensitivity_parameters():
    sns.set_palette("Dark2")
    fig, axs = plt.subplots(2, 4, figsize=(TEXTWIDTH, TEXTWIDTH/2), dpi=200)

    # p_gen
    keyword = {"remark": "parameter_regime p_gen"}
    parameters = find_record_patterns(keyword)
    ID = parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "full_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.00035379511745899935, 0.0016242539266574066, 0.00341351572896402, 0.0055371405771855895, 0.007951730651503517])
    key_rate_list = np.array(key_rate_list)
    axs[1][0].plot(parameters["p_gen"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_with_approx_tau, '*', label="non-uniform")
    axs[0][0].plot(parameters["p_gen"], key_rate_list * 1.0e3, '*', label="non-uniform")

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.0003196067521768056, 0.0015952289413017596, 0.0034020110572561056, 0.005537270040690616, 0.00795739066267918])
    key_rate_list = np.array(key_rate_list)
    axs[1][0].plot(parameters["p_gen"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_with_approx_tau, '+', label="uniform")
    axs[0][0].plot(parameters["p_gen"], key_rate_list * 1.0e3, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["p_gen"]))
    for i, p_gen in enumerate(parameters["p_gen"]):
        key = (p_gen, parameters["p_swap"], parameters["w0"], parameters["t_coh"], "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][0].plot(parameters["p_gen"], no_timeout_key_rate_list * 1.0e3, '.', label="no tau")

    axs[0][0].set_xticklabels([])
    axs[0][0].set_ylabel(r"$R(\tau_{\rm{target}}) \quad (10^{-3})$")
    axs[0][0].text(0.5, 0., "(a)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][0].text(0.5, 0.9, "(e)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][0].plot(0.1, 0, '.', label="No cut-off")
    axs[1][0].plot(0.1, 0, 'o')
    axs[1][0].set_ylim((-0.05, 1.05))
    axs[1][0].set_ylabel(r"Relative $R$ improvement of"+"\n"+r"$\tau_{\rm{target}}$ vs. $\tau_{\rm{baseline}}$")
    axs[1][0].legend(fontsize="x-small", loc = 2)
    axs[1][0].set_xlabel(r"$p_{\rm{gen}}$")
    # axs[0][0].legend()

    # p_swap
    keyword = {"remark": "parameter_regime p_swap"}
    parameters = find_record_patterns(keyword)
    ID = parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "full_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.0001219172172505286,0.0003538233527971525, 0.0008003625102358013, 0.0015626213740068805, 0.0027620078708939086, 0.004545283797514378])
    key_rate_list = np.array(key_rate_list)
    axs[1][1].plot(parameters["p_swap"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_with_approx_tau, '*', label="non-uniform")
    axs[0][1].plot(parameters["p_swap"], key_rate_list * 1.0e3, '*', label="non-uniform")

    tau_list = []
    key_rate_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.00010577463895730655, 0.0003197017677312484, 0.0007438540150383622, 0.0014865613177057484, 0.002677938426405697, 0.004477177475977771])
    key_rate_list = np.array(key_rate_list)
    axs[1][1].plot(parameters["p_swap"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_list, '+', label="uniform")
    axs[0][1].plot(parameters["p_swap"], key_rate_list * 1.0e3, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["p_swap"]))
    for i, p_swap in enumerate(parameters["p_swap"]):
        key = (parameters["p_gen"], p_swap, parameters["w0"], parameters["t_coh"], "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][1].plot(parameters["p_swap"], no_timeout_key_rate_list * 1.0e3, '.', label="no tau")

    axs[1][1].plot(0.5, 0, '.')
    axs[1][1].plot(0.5, 0, 'o')
    axs[1][1].set_ylim((-0.05, 1.05))
    # axs[1][1].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][1].legend()
    axs[0][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel(r"$p_{\rm{swap}}$")
    # axs[0][1].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][1].legend()
    axs[0][1].text(0.9, 0., "(b)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][1].text(0.9, 0.9, "(f)", horizontalalignment='right', verticalalignment='bottom')

    # w0
    keyword = {"remark": "parameter_regime w0 new"}
    parameters = find_record_patterns(keyword)
    ID= parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "full_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array(
        [
            0.0, 3.654452057601761e-05, 0.00010216327562239115, 
            0.00015259540087151008, 0.00021043602342102674, 0.0002899371117830144,
            0.0003537917531146599, 0.0004217177505176499, 0.0005190238209611317,
            0.0006000531872573797, 0.0006815169130366861, 0.0008026348162433884,
            0.0009066309267816245])
    key_rate_list = np.array(key_rate_list)
    axs[1][2].plot(parameters["w0"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_list, '*', label="non-uniform")
    axs[0][2].plot(parameters["w0"], key_rate_list * 1.0e3, '*', label="non-uniform")
    
    tau_list = []
    key_rate_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array(
        [0.0, 1.5434732484909487e-05, 7.845444363280457e-05,
        0.00012741982175137038, 0.00018237547932027607, 0.0002586269900084511,
        0.0003196067521768056, 0.0003849087408666171, 0.0004780543102126605,
        0.0005544076944605199, 0.0006333803939740579, 0.000748948383868109,
        0.0008458337250658964])
    axs[1][2].plot(parameters["w0"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_list, '+', label="uniform")
    axs[0][2].plot(parameters["w0"], key_rate_list * 1.0e3, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["w0"]))
    for i, w0 in enumerate(parameters["w0"]):
        key = (parameters["p_gen"], parameters["p_swap"], w0, parameters["t_coh"], "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][2].plot(parameters["w0"], no_timeout_key_rate_list * 1.0e3, '.', label="no tau")

    axs[1][2].plot(0.98, 0, '.')
    axs[1][2].plot(0.98, 0, 'o')
    axs[1][2].set_ylim((-0.05, 1.05))
    # axs[1][2].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][2].legend()
    axs[0][2].set_xticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][2].set_xlabel(r"$w_{\rm{0}}$")
    # axs[0][2].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][2].legend()
    axs[0][2].text(1., 0., "(c)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][2].text(1., 0.9, "(g)", horizontalalignment='right', verticalalignment='bottom')

    # t_coh
    keyword = {"remark": "parameter_regime t_coh"}
    parameters = find_record_patterns(keyword)
    ID= parameters["ID"]
    data_dict = load_data(ID)

    tau_list = []
    key_rate_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "full_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.0, 8.988723166647054e-05, 0.00019783240294677181, 0.0002838314598252032, 0.00035383778746811293, 0.0004118821138325925, 0.00046076606556231267, 0.0005385103833126262, 0.0005975548089184896, 0.0006439139763454823, 0.0007120311723360387, 0.0008092243210318132])
    key_rate_list = np.array(key_rate_list)
    axs[1][3].plot(parameters["t_coh"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_list, '*', label="non-uniform")
    axs[0][3].plot(parameters["t_coh"], key_rate_list * 1.0e3, '*', label="non-uniform")
    
    tau_list = []
    key_rate_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "uniform_de")
        tau = data_dict[key]["tau"]
        tau_list.append(tau)
        pmf = data_dict[key]["pmf"]
        key_rate_list[i] = data_dict[key]["key_rate"]
    key_rate_with_approx_tau = np.array([0.0, 4.9269527234188307e-05, 0.00015930936237654582, 0.0002475408358425697, 0.0003197136748064556, 0.00037978309712547646, 0.00043052911783455437, 0.0005115227169767813, 0.0005732663901512737, 0.0006218828033737355, 0.0006935344802208565, 0.0007962167861329947])
    axs[1][3].plot(parameters["t_coh"], -(key_rate_with_approx_tau-key_rate_list)/key_rate_list, '+', label="uniform")
    axs[0][3].plot(parameters["t_coh"], key_rate_list * 1.0e3, '+', label="uniform")

    no_timeout_key_rate_list = np.empty(len(parameters["t_coh"]))
    for i, t_coh in enumerate(parameters["t_coh"]):
        key = (parameters["p_gen"], parameters["p_swap"], parameters["w0"], t_coh, "none")
        tau = data_dict[key]["tau"]
        pmf = data_dict[key]["pmf"]
        no_timeout_key_rate_list[i] = data_dict[key]["key_rate"]
    no_timeout_key_rate_list = np.array(no_timeout_key_rate_list)
    axs[0][3].plot(parameters["t_coh"], no_timeout_key_rate_list * 1.0e3, '.', label="no tau")

    axs[1][3].plot(400, 0, '.')
    axs[1][3].plot(400, 0, 'o')
    axs[1][3].set_ylim((-0.05, 1.05))
    # axs[1][3].set_ylabel(r"$(R_0-R_{\rm{target}})/R_{\rm{target}}$")
    # axs[1][3].legend()
    axs[0][3].set_xticklabels([])
    axs[1][3].set_yticklabels([])
    axs[1][3].set_xlabel(r"$t_{\rm{coh}}$")
    # axs[0][3].set_ylabel(r"$R_{\rm{target}}$")
    # axs[0][3].legend()
    axs[0][3].text(1500, 0., "(d)", horizontalalignment='right', verticalalignment='bottom')
    axs[1][3].text(1500, 0.9, "(h)", horizontalalignment='right', verticalalignment='bottom')

    plt.subplots_adjust(top=0.98, bottom=0.1, right=0.98, left=0.09)
    fig.savefig("figures\\tau_sensitivity.pdf")
    fig.savefig("figures\\tau_sensitivity.png")
    return fig


plt.rcParams.update(
    {"font.size": 9,
    "font.family": "Time",
    "text.usetex": False,
    # 'legend.fontsize': 'x-large',
    'axes.labelsize': 'small',
    # 'axes.titlesize':'x-small',
    'xtick.labelsize':'x-small',
    'ytick.labelsize':'x-small'
    })

if __name__ == "__main__":
    # fig 4
    # plot_swap_with_cutoff_data()
    plot_swap_with_cutoff_fig()

    # fig 5
    # fig = plot_trade_off_data()
    fig = plot_trade_off_fig()

    # fig 6
    plot_sensitivity_parameters()

    # fig 7
    plot_parameter_contour()

    plt.show()

    # example of obtaining data for fig 6 and fig 7:
    # # Those parameter in a list will be iterated for all combination, in this example, you will get a grid for all w0 and t_coh combinition, and also for uniform and nonuniform cutoff

    # if __name__ =="__main__":
    #     parameters = {"protocol": (0, ),
    #         "p_gen": 0.1,
    #         "p_swap": 0.5,
    #         "t_trunc": 200,
    #         "w0": [0.97, 0.98, 0.99, 1.0],
    #         "t_coh": [300, 350, 400, 450, 500, 550, 600, 700, 800, 1000, 1200],
    #         "optimizer": ["full_de", "uniform_de", "none"],
    #         }
    #     ID = parameter_regime(parameters, remark="example run")
