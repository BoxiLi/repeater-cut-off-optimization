from copy import deepcopy
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from optimize_cutoff import (
    optimization_tau_wrapper,
    CutoffOptimizer,
    full_tau_pretrain)
from utility_functions import secret_key_rate, werner_to_fid, get_mean_waiting_time, get_mean_werner
from logging_utilities import log_init, log_params, log_finish, printProgressBar, save_data, load_data, find_record_id, find_record_patterns, create_iter_kwargs
from repeater_algorithm import repeater_sim, plot_algorithm, RepeaterChainSimulation
from repeater_mc import repeater_mc, plot_mc_simulation
from plot_paper import parameter_regime, complete_data


TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487
colors = ["#9b59b6", "#3498db", "#95a5a6", "teal", "#e74c3c", "#2ecc71"]

def compare_distribution():
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.4,
        "w0": 0.98,
        "t_coh": 600,
        "t_trunc": 10000,
        "cut_type": ["memory_time", "fidelity", "run_time"],
        "mt_cut": (23, 47, 87),
        "rt_cut": (33, 70, 114),
        "w_cut": (0.93980861, 0.87538725, 0.75263336),
        }
    t_trunc = 1800
    fig = plt.figure(figsize=(TEXTWIDTH, LINEWIDTH/1.5), dpi=250)

    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.1, hspace=0.1)

    kwarg_list = create_iter_kwargs(parameters)
    axis1 = []
    axis2 = []
    for i, kwarg in enumerate(kwarg_list):
        y1max = 0.
        y1min = np.inf
        y2max = 0.
        y2min = 1

        ax1 = plt.subplot(gs[0+i])
        axis1.append(ax1)
        ax2 = plt.subplot(gs[3+i])
        axis2.append(ax2)
        pmf, w_func = repeater_sim(kwarg)
        print(get_mean_waiting_time(pmf))
        print(werner_to_fid(get_mean_werner(pmf, w_func)))
        print(secret_key_rate(pmf, w_func))
        t = 0
        while(pmf[t] < 1.0e-17):
            w_func[t] = np.nan
            t += 1
        ax1.plot(100*pmf[:t_trunc], label=kwarg["cut_type"], linewidth=1, color="slategrey")
        rate = secret_key_rate(pmf, werner_to_fid(w_func))
        ax2.plot(w_func[:t_trunc], label=str(rate), linewidth=1, color="slategrey")

        y1max = max(ax1.get_ylim()[1], y1max)
        y1min = min(ax1.get_ylim()[0], y1min)
        y2max = max(ax2.get_ylim()[1], y2max)
        y2min = min(ax2.get_ylim()[0], y2min)

        if i == 0.:
            ax1.set_ylabel(r"$T$ $(10^{-2})$")
            ax2.set_ylabel(r"Fidelity $F(t)$")
        else:
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        if i == 1:
            ax2.set_xlabel(r"Waiting time $T$")
        title = {0:"DIF-TIME-CUT-OFF", 1:"FIDELITY-CUT-OFF", 2:"MAX-TIME-CUT-OFF"}
        ax1.set_title(title[i], fontsize=7)

    for i in range(3):
        ax1 = axis1[i]
        ax2 = axis2[i]
        ax1.set_ylim((min(y1min, ax1.get_ylim()[0]), max(y1max, ax1.get_ylim()[1])))
        ax2.set_ylim((min(y2min, ax2.get_ylim()[0]), max(y2max, ax2.get_ylim()[1])))

    mean_waiting_time = [819, 795, 960]
    mean_fid = [0.786, 0.784, 0.790]
    secret_key = [2.207, 2.211, 1.979]
    for i in range(3):
        ax2 = axis2[i]
        ax2.text(1000, 0.82, r"$\bar{T}=$"+f"{mean_waiting_time[i]}", fontsize=6)
        ax2.text(1000, 0.79, r"$\bar{F}=$"+f"{mean_fid[i]}", fontsize=6)
        ax2.text(1000, 0.76, r"$R=$"+f"{secret_key[i]}"+r"$\cdot $10$^{-4}$", fontsize=6)

    fig.subplots_adjust(bottom=0.15, left = 0.07, right = 0.99, top = 0.90)
    fig.tight_layout()
    fig.savefig("figures/comparison.pdf")
    fig.savefig("figures/comparison.png")
    fig.show()
    input()
        

# def cutoff_vs_coherence_data():
#     parameters = {
#         "protocol": (0, 0, 0),
#         "p_gen": 0.1,
#         "p_swap": 0.8,
#         "t_trunc": 2000,
#         "w0": 0.98,
#         "t_coh": [160, 180, 200, 220, 240, 260, 280, 300, 330, 360, 400, 430, 460, 500, 550, 600, 650, 700, 750, 800],
#         "optimizer": ["nonuniform_de", "none"],
#         "cut_type": "memory_time",
#         "sample_distance": 1,
#         }

#     ID = log_init("optimize", level=logging.EXP)
#     parameter_regime(parameters, ID, remark="compare cutoff type t_coh")
#     complete_data(ID)
#     return ID

def cutoff_vs_coherence_plot(ID):
    parameters = find_record_id(ID)
    data_dict = load_data(ID)

    del_t_coh_num = 3
    t_coh_list = np.asarray(parameters["t_coh"][:-del_t_coh_num])
    cutoff_list = []
    optimizer = "nonuniform_de"
    if isinstance(parameters["w0"], list):
        w0 = parameters["w0"][3]
    else:
        w0 = parameters["w0"]
    cut_off_list = []
    cut_type = "memory_time"

    upper_bound = []
    lower_bound = []
    for i, t_coh in enumerate(t_coh_list):
        key = (parameters["p_gen"], parameters["p_swap"], w0, t_coh, optimizer)
        best_key_rate_rate = data_dict[key]["key_rate"]
        cutoff_dict = data_dict[key]["tau"]
        cutoff_list.append(cutoff_dict[cut_type])

    #     # get the error bound
    #     level = 1
    #     kwargs = deepcopy(parameters)
    #     kwargs["t_coh"] = t_coh
    #     kwargs["w0"] = w0
    #     kwargs["cut_type"] = "memory_time"
    #     mt_cut1 = deepcopy(cutoff_dict["memory_time"])
    #     mt_cut2 = deepcopy(cutoff_dict["memory_time"])
    #     new_secret_key_rate = best_key_rate_rate
    #     while abs(new_secret_key_rate - best_key_rate_rate)/best_key_rate_rate < 0.01:
    #         print(mt_cut1)
    #         mt_cut1[level] += 1
    #         kwargs["mt_cut"] = mt_cut1
    #         pmf, w_func = repeater_sim(kwargs)
    #         new_secret_key_rate = secret_key_rate(pmf, w_func)
    #     new_secret_key_rate = best_key_rate_rate
    #     upper_bound.append(mt_cut1[level] - cutoff_dict["memory_time"][level])
    #     while abs(new_secret_key_rate - best_key_rate_rate)/best_key_rate_rate < 0.01:
    #         print(mt_cut2)
    #         mt_cut2[level] -= 1
    #         kwargs["mt_cut"] = mt_cut2
    #         pmf, w_func = repeater_sim(kwargs)
    #         new_secret_key_rate = secret_key_rate(pmf, w_func)
    #     lower_bound.append(cutoff_dict["memory_time"][level] - mt_cut2[level])
    # print(upper_bound)
    # print(lower_bound)

    fig, ax = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH/1.5), dpi=250)
    cutoff_list = np.transpose(np.asarray(cutoff_list))
    # lower_bound=[
    #     [3, 3, 3, 4, 4, 5, 5, 4, 5, 5, 6, 6, 7, 9],
    #     [4, 5, 5, 5, 6, 7, 6, 7, 8, 9, 9, 10, 11, 13],
    #     [7, 7, 8, 8, 9, 10, 10, 12, 13, 13, 14, 14, 16, 19],
    # ]
    # upper_bound=[
    #     [3, 4, 4, 4, 5, 5, 6, 8, 8, 9, 10, 12, 13, 28],
    #     [5, 5, 5, 7, 7, 7, 9, 9, 10, 11, 12, 13, 14, 20],
    #     [7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 18, 18, 23],
    # ]
    for i in range(len(parameters["protocol"])):
        ax.errorbar(t_coh_list/10000, cutoff_list[i]/10000, fmt=".", capsize=2, color=colors[i])

    ax.set_xlabel(r"Memory coherence time $t_{\rm{coh}} \quad (10^4)$")
    ax.set_ylabel(r"optimal cut-off $\tau \quad (10^4)$")
    
    ax2 = ax.twinx()
    def func(x, a, b):
        return a * x + b
    for i in range(len(parameters["protocol"])):
        par, cov = curve_fit(func, t_coh_list, cutoff_list[i])
        ax2.plot(t_coh_list/10000, func(t_coh_list, *par)/10000, color=colors[i], label=f"level={i}")
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        ax2.legend()
    fig.subplots_adjust(bottom=0.18, left = 0.13, right = 0.95, top = 0.98)
    fig.savefig("figures/linear.pdf")
    fig.savefig("figures/linear.png")
    fig.show()
    input()

if __name__ == "__main__":

    plt.rcParams.update(
        {"font.size": 9,
        # "font.family": "Arial",
        'legend.fontsize': 'x-small',
        'axes.labelsize': 'small',
        # 'axes.titlesize':'x-small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small',
        })

    # ID = cutoff_vs_coherence_data()
    # cutoff_vs_coherence_plot("optimize-20200915-150426")

    # compare_distribution()

    # set parameters
    # log_init("plot", level=logging.ERROR)

    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 1.0,
        "t_coh": 500,
        "t_trunc": 10000,
        "cut_type": "memory_time",
        "tau": 42,
        # "w_cut": (0.90600844, 0.84714154, 0.75991523),
        # "sample_distance": 50
        }

    # import time
    # simulator = RepeaterChainSimulation()
    # simulator.nested_protocol(parameters=parameters)

    # result = []
    # error = []
    # if parameters["p_swap"] == 0.5:
    #     trunc_list = [356, 1059, 3248, 10394, 35264, 129588, 531609, 2534515]
    # if parameters["p_swap"] == 0.9:
    #     trunc_list = [81, 118, 179, 286, 504, 1062, 1377, 3764]
    # if parameters["p_swap"] == 0.2:
    #     trunc_list = [2465, 18918, 149276, 1217651]
    # try:
    #     max_length = len(trunc_list)
    # except:
    #     max_length = 8
    # for iter, scale in enumerate(range(1, max_length+1)):
    #     kwargs = deepcopy(parameters)
    #     # kwargs["p_gen"] /= scale
    #     p_swap = parameters["p_swap"]
    #     kwargs["protocol"] = (0,) * scale + (0,)
    #     kwargs["t_coh"] *= 1/p_swap**scale
    #     try:
    #         kwargs["t_trunc"] = trunc_list[iter]
    #     except:
    #         kwargs["t_trunc"] *= int(np.ceil(2/p_swap**scale))
    #     kwargs["tau"] *= int(np.ceil(1/p_swap**scale))

    #     print(kwargs)

    #     temp_list = []
    #     for i in range(10):

        
    #         start_time = time.time()
    #         simulator = RepeaterChainSimulation()
    #         pmf, w_func = simulator.nested_protocol(parameters=kwargs)
    #         end_time = time.time()
    #         cdf = np.cumsum(pmf)
    #         try:
    #             trunc_list
    #         except:
    #             print(next(i for i,v in enumerate(cdf) if v > 0.99))
    #         temp_list.append(end_time - start_time)
        
    #     result.append(np.mean(temp_list))
    #     error.append(np.std(temp_list)/np.sqrt(len(temp_list)))
    #     print(result)
    # print(result)
    # print(error)

    trunc = (np.array([1,2,3,4,5,6, 7, 8])+1)
    run_time_05 = [0.0012997865676879882, 0.003989267349243164, 0.00857694149017334, 0.042984938621520995, 0.37748136520385744, 0.9196482419967651, 10.277681493759156, 48.4191987991333]
    run_time_09 = [0.0008984088897705078, 0.001595751444498698, 0.002094451586405436, 0.0029919942220052085, 0.003723438580830892, 0.008245484034220377, 0.009474841753641765, 0.019250933329264322]
    run_time_02 = [0.010637601216634115, 0.10438728332519531, 
    1.2382001876831055, 13.413824955622355]
    fig, ax = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH/1.7), dpi=250)
    ax.plot([2**n+1 for n in range(2, len(run_time_02)+2)], run_time_02, '*', label=r"$p_{\rm{swap}=0.2}$", color=colors[0])
    ax.plot([2**n+1 for n in range(2, len(run_time_05)+2)], run_time_05, '.', label=r"$p_{\rm{swap}=0.5}$", color=colors[1])
    ax.plot([2**n+1 for n in range(2, len(run_time_09)+2)], run_time_09, '+', label=r"$p_{\rm{swap}=0.9}$", color=colors[3])
    # ax.plot(trunc, run_time_02, '+', label=r"$p_\rm{swap}=0.2$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Computation time (s)")
    # ax.set_title("computation time" + r"$\propto$" + "#nodes", fontsize="small")
    ax.legend(fontsize = "x-small")
    ax.set_xticks([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
    ax.set_xticklabels([f"{2**2+1}",f"{2**3+1}",f"{2**4+1}",f"{2**5+1}",f"{2**6+1}",f"{2**7+1}",f"{2**8+1}",f"{2**9+1}"])
    # remove minor ticks
    import matplotlib
    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.set_xlim((0.5, 8.5))
    # ax.set_ylim((0.01, 300))
    ax.grid("True")
    fig.tight_layout()
    fig.show()
    fig.savefig("figures/computation_time.pdf")
    fig.savefig("figures/computation_time.png")
 
