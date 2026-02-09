#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to varying ploting functions.
"""

import os
import numpy as np
from copy import deepcopy
from pylab import *
from typing import (Sequence, List)
from matplotlib import pyplot as plt
import statsmodels.api as sm

__all__ = [
    "plot_cluster_thr",
]


""" 
Policy labels.
"""
policy_labels = {
    "crius": "Crius",
    "crius-backup": "Crius",
    "crius-dp": "Crius-DP",
    "fcfs": "FCFS",
    "fcfs-r": "FCFS-RE",
    "gandiva": "Gandiva",
    "elasticflow": "ElasticFlow",
    "elasticflow-l": "ElasticFlow-LS",
    "elasticflow-alpa": "ElasticFlow-Alpa",
    "gavel": "Gavel",
    "sia": "Sia",
}

policy_labels_small = {
    "crius": "Crius",
    "fcfs": "FCFS",
    "fcfs-r": "FCFS-RE",
    "gandiva": "Gandiva",
    "elasticflow": "EF",
    "elasticflow-l": "EF-LS",
    "elasticflow-alpa": "EF-A",
    "gavel": "Gavel",
}

# policies = ["fcfs", "gandiva", "elasticflow", "elasticflow-l", "gavel", "crius-ddl", "crius-ns", "crius-nh", "crius-backup", "crius"]
policies = ["fcfs", "elasticflow-l", "gavel", "sia", "crius"]
# policies_small = ["fcfs", "gandiva", "elasticflow-l", "gavel", "crius"]
policies_all = ["fcfs", "gandiva", "elasticflow", "elasticflow-l", "sia", "gavel", "crius-ddl", "crius-ns", "crius-nh", "crius-dp", "crius"]

# Availble color
# avail_color = ["#434475", "#87BCBD", "#969BC7"]
# avail_color = [ "#469EB4", "#87CFA4", "#FDB96A", "#F57547", "#4E62AB" ]
avail_color = [ "#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#F57547", "#4E62AB" ]
# Line style
line_style_list = [ ":", "-.", "--", "--", "--", "-" ]


#################################
#    Plot Cluster Throughput    #
#################################

def plot_cluster_thr(
    work_dir: str, 
    out_dir: str = "./figures",
    trace_type: str = "philly", 
    only_cal: bool = False,
    all_policies: bool = False,
) -> None:
    """
    Plot the cluster throughput throughout the entire trace.
    """

    # Load data
    thr_data = _read_thr_data(work_dir, all_policies)

    # Canvas
    if trace_type == "philly":
        # fig, ax1 = plt.subplots(figsize = (18, 2.5), nrows = 1, dpi = 400)
        # Single column
        fig, ax1 = plt.subplots(figsize = (18, 4), nrows = 1, dpi = 400)
    elif trace_type == "helios" or trace_type == "pai":
        fig, ax1 = plt.subplots(figsize = (8, 6), nrows = 1, dpi = 400)
    subplots_adjust(left=0, bottom=0, top=1.0, right=1, hspace=0.3)
    
    # Thr stats
    thr_threshold = 100
    crius_avg_thr = np.mean(
        [_elm for _elm in thr_data[-1][1] if _elm > thr_threshold]
    )
    for rec in thr_data:
        thrs = [_elm for _elm in rec[1] if _elm > thr_threshold]
        print(f"[I] Policy: {rec[0]} | Average cluster throughput: {np.mean(thrs)} | " + 
              f"Peak cluster throughput: {np.max(thrs)} | P99 cluster throughput" + 
              f": {np.percentile(thrs, 99)}")

        if rec[0] == "crius":
            continue

        # Improvement ratio
        _ratio = round(crius_avg_thr / np.mean(thrs), 3)
        print("[I] Improvement ratio of crius on {}: {}".format(rec[0], _ratio))

    if only_cal:
        # Skip plot
        return 
    
    # Data flatten
    frame_size = 5 if trace_type != "pai" else 7
    flatten_style = "mean"
    _thr_data = list()
    for rec in thr_data:
        _thr_data.append(
            [rec[0], _flatten_curve(rec[1], frame_size, flatten_style)]
        )
    thr_data = deepcopy(_thr_data)
    
    # "#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#4E62AB"
    # # Colors
    # color_table = {
    #     # "fcfs": "#469EB4",
    #     # "gandiva": "#87CFA4",
    #     # "elasticflow-l": "#FDB96A",
    #     # "gavel": "#F57547",
    #     # "crius": "#4E62AB",
    #     "fcfs": "#5260A5",
    #     "gandiva": "#6EC8C1",
    #     "elasticflow-l": "#A6B7DE",
    #     "gavel": "dimgray",
    #     "crius": "orange",
    # } if trace_type == "philly" else {
    #     "fcfs": "#8ECFC9",
    #     "gandiva": "#FFBE7A",
    #     "elasticflow-l": "lightgray",
    #     "gavel": "#acc2d9",
    #     "crius": "#82B0D2",
    # }

    # # Colors
    # color_table = {
    #     "fcfs": '#8DC6C2',
    #     "gandiva": '#EE7B68',
    #     "elasticflow-l": '#F8B878',
    #     "gavel": '#7CA6C9',
    #     "sia": '#acc2d9',
    #     "crius": '#2A6CA6',
    # }

    # Colors
    color_table = {
        "fcfs": '#8E7BA0',
        "gandiva": '',
        "elasticflow-l": '#EE8636',
        "gavel": '#549E3F',
        "sia": '#666666',
        "crius-dp": 'black',
        "crius": '#3C76AF',
    }

    # Line styles
    ls_table = {
        "fcfs": ":",
        "gandiva": "-.",
        "elasticflow-l": "--",
        "gavel": "--",
        "sia": "--",
        "crius-dp": "-",
        "crius": "-",
    }
    
    # Plot
    linewidth = 3.5 if trace_type == "philly" else 3.0
    for i, data in enumerate(thr_data):
        ax1.plot(data[1], 
                 color=color_table[data[0]], 
                 label=policy_labels[data[0]], 
                 linestyle=ls_table[data[0]], 
                 linewidth=linewidth)

    # Grid
    ax1.grid(axis="y", color="lightgray", linestyle="--", zorder=0)

    # # Legends
    # ax1.legend(ncol=1,
    #            loc='upper right',
    #            fontsize=16,
    #            markerscale=3,
    #            labelspacing=0.1,
    #            edgecolor='black',
    #            shadow=False,
    #            fancybox=False,
    #            handlelength=0.8,
    #            handletextpad=0.6,
    #            columnspacing=0.6,
    #            borderaxespad=0.5,
    #         )
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=18)
    
    if trace_type == "philly":    
        # Y axis
        # y_ticks = [0.0, 2000, 4000, 6000, 8000, 10000, 12000]
        y_ticks = [0.0, 4000, 8000, 12000]
        # y_labels = ["0", "4k", "8k", "12k", "16k", "20k"]
        # y_labels = ["", "", "", "", "", "", "", ""]
        y_labels = ["", "", "", ""]
        plt.yticks(y_ticks, y_labels, fontsize=19, weight="bold")
        plt.ylim(0.0, 12000.0)
        plt.ylabel("", fontsize=22, weight="bold")

        # X axis
        x_ticks = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        # plt.xticks(x_ticks, x_labels, fontsize=19, weight="bold")
        plt.xticks(x_ticks, labels="", fontsize=19, weight="bold")
        plt.xlim(0.0, 2000.0)
    elif trace_type == "helios":
        # Y axis
        y_ticks = [0.0, 1000, 2000, 3000, 4000, 5000]
        # y_labels = ["0", "4k", "8k", "12k", "16k", "20k"]
        y_labels = ["", "", "", "", "", ""]
        plt.yticks(y_ticks, y_labels, fontsize=19, weight="bold")
        plt.ylim(0.0, 5000.0)
        plt.ylabel("", fontsize=22, weight="bold")

        # X axis
        x_ticks = [0, 100, 200, 300, 400, 500]
        # plt.xticks(x_ticks, x_labels, fontsize=19, weight="bold")
        plt.xticks(x_ticks, labels="", fontsize=19, weight="bold")
        plt.xlim(0.0, 500.0)
    elif trace_type == "pai":
        # Y axis
        y_ticks = [0.0, 300, 600, 900, 1200, 1500]
        # y_labels = ["0", "4k", "8k", "12k", "16k", "20k"]
        y_labels = ["", "", "", "", "", ""]
        plt.yticks(y_ticks, y_labels, fontsize=19, weight="bold")
        plt.ylim(0.0, 1500.0)
        plt.ylabel("", fontsize=22, weight="bold")

        # X axis
        x_ticks = [0, 100, 200, 300, 400, 500]
        # plt.xticks(x_ticks, x_labels, fontsize=19, weight="bold")
        plt.xticks(x_ticks, labels="", fontsize=19, weight="bold")
        plt.xlim(0.0, 500.0)
    else:
        raise ValueError(f"Invalid trace type: {trace_type}")

    # Save
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    suffix = "" if trace_type == "philly" else f"_{trace_type}"
    plt.savefig(f"{out_dir}/cluster_thr{suffix}.pdf", bbox_inches="tight")


def _read_thr_data(work_dir: str, all_policies: bool = False):
    """ Read all thr data under the work dir. """
    
    thr_data = list()
    file_names = os.listdir(work_dir)
    used_policies = policies if not all_policies else policies_all
    
    for file_name in file_names:
        strs = file_name.split("_")
        if (len(strs) != 2 or 
            (len(strs) == 2 and strs[1] != "thr.npy")):
            continue
        
        policy = strs[0]
        if policy not in used_policies:
            continue
        
        thr_data.append([
            policy, 
            list(np.load(os.path.join(work_dir, file_name))),
        ])
    
    for rec in thr_data:
        print(f"[I] Policy: {rec[0]} | Average thr: {np.mean(rec[1])} | Max thr: {np.max(rec[1])}")

    exit(0)
    
    return sorted(thr_data, key=lambda x: used_policies.index(x[0]))


#################################
#       Plot Queuing Time       #
#################################

def plot_queuing_time(work_dir: str, trace_type: str = "philly", only_cal: bool = False):
    """
    Plot the average job queuing time.
    """
    print("[I] Analyzing queuing time of all policies and plot...")

    if trace_type != "philly":
        return _plot_queuing_time_venus_pai(work_dir, trace_type)
    
    # Read queuing time data
    _queuing_time_data = _read_queuing_time_data(work_dir)
    _queuing_time_data = sorted(_queuing_time_data, key=lambda x: policies_all.index(x[0]))
    
    policy_names = [_d[0] for _d in _queuing_time_data]
    queuing_time_data_raw = [np.mean(_item[1]) for _item in _queuing_time_data]
    # Normalize
    _max = np.max(queuing_time_data_raw)
    # _max = np.sort(queuing_time_data)[-2]
    queuing_time_data = [round(_v / _max, 3) for _v in queuing_time_data_raw]

    for _i, _policy_name in enumerate(policy_names):
        print(f"[I] Policy name: {_policy_name} | Raw queuing time: {queuing_time_data_raw[_i]} | Normalized queuing time: {queuing_time_data[_i]}")

    if only_cal:
        return 
    
    # Canvas
    fig, (ax1) = plt.subplots(figsize = (8, 5), nrows = 1, dpi = 400)
    fig.tight_layout(pad=5.0) 
    subplots_adjust(left=0, bottom=0.0, top=1.0, right=1, hspace=0.7)

    # Group num
    x = np.arange(len(queuing_time_data))

    # Size, n denotes category num
    total_width, n = 0.8, 1
    width = total_width / n
    x = x + 1.25 * width

    _data = [min(1.1, _val) for _val in queuing_time_data]
    
    for _idx in range(len(queuing_time_data)):
        # Plot
        ax1.bar(x, _data, width=width * 0.7, color=avail_color[:len(_data)], edgecolor="black", zorder=10)
        # Annotations 
        for idx, y_value in enumerate(_data):
            ax1.text(idx + 1.0, y_value + 0.03, queuing_time_data[idx], fontsize=19, weight="bold", horizontalalignment="center")
    # Grid
    ax1.grid(axis="y", color="gray", zorder=0)
    
    # X axis
    plt.xticks(np.array([1, 2, 3, 4, 5]), [policy_labels[_p] for _p in policy_names], fontsize=19, weight="bold", rotation=15)
    plt.xlim(0.5, 5.5)
    plt.xlabel("Policies", fontsize=22, weight="bold")

    # Y axis
    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(y_ticks, fontsize=19, weight="bold")
    plt.ylim(0.0, 1.2)
    plt.ylabel("Normalized Queuing Time", fontsize=22, weight="bold")

    # Path
    file_path = "./figures/queuing_time.pdf"
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def _plot_queuing_time_venus_pai(work_dir: str, trace_type: str):
    """ Plot job queuing time for Helios Venus and PAI traces. """
    # Read queuing time data
    _queuing_time_data = _read_queuing_time_data(work_dir)
    queuing_time_data = sorted(_queuing_time_data, key=lambda x: policies.index(x[0]))
    queuing_time_stats = dict()
    for _rec in queuing_time_data:
        queuing_time_stats[_rec[0]] = [
            np.mean(_rec[1]), np.percentile(_rec[1], 99), np.percentile(_rec[1], 70), np.max(_rec[1]), np.min(_rec[1])
        ] 

    print(f"[I] Trace type: {trace_type}.")
    for _policy_name in queuing_time_stats:
        (_mean, _99_per, _70_per, _max, _min) = queuing_time_stats[_policy_name]
        print(f"[I] Policy name: {_policy_name} | Mean queuing time: {_mean} s | " + 
              f"Percentile-99 time: {_99_per} s | Percentile-70 time: {_70_per} s | " + 
              f"Max queuing time: {_max} s | Min queuing time: {_min} s.")


def _read_queuing_time_data(work_dir: str):
    """ Read all thr data under the work dir. """
    queuing_time_data = list()
    file_names = os.listdir(work_dir)
    for file_name in file_names:
        if (len(file_name.split("_")) != 3) or (len(file_name.split("_")) == 3 and file_name.split("_")[1] != "queuing"):
            continue
        _policy = file_name.split("_")[0]
        if _policy not in policies:
            continue
        _file_path = work_dir + "/" + file_name
        # Load
        _queuing_time_data = list(np.load(_file_path))
        queuing_time_data.append([_policy, _queuing_time_data])
    return queuing_time_data


#################################
#    Plot Job Completion Time   #
#################################

def read_jct_data_and_plot(
    work_dir: str, 
    trace_type: str = "philly", 
    all_policies: bool = False,
    only_ddl: bool = False,
):
    """ Read all JCT data under the work dir. """
    print("[I] Analyzing JCT of all policies and plot...")
    
    jct_data = list()
    crius_avg_jct, max_finished_job_num, max_jct = None, None, 0
    used_policies = policies if not all_policies else policies_all
    # if only_ddl:
    #     used_policies = ["crius-ddl", "elasticflow"]
    file_names = os.listdir(work_dir)
    
    for file_name in file_names:
        if (len(file_name.split("_")) != 2) or (len(file_name.split("_")) == 2 and file_name.split("_")[1] != "jct.npy"):
            continue
        _policy = file_name.split("_")[0]
        if _policy not in used_policies:
            continue
        _file_path = work_dir + "/" + file_name

        print(_file_path)

        _jct_data = list(np.load(_file_path))
        
        if (_policy == "crius" or (only_ddl and _policy == "crius-ddl")):
            crius_avg_jct = np.mean(_jct_data)
            max_finished_job_num = len(_jct_data)

        max_jct = max(np.max(_jct_data), max_jct)
        jct_data.append([_policy, _jct_data])

        print(_policy, len(_jct_data))
    
    for _rec in jct_data:
        print(f"[I] Policy: {_rec[0]} | Avergae JCT of finished jobs: {np.mean(_rec[1])}")
    
    if max_finished_job_num is None:
        return

    # Calculate improvement of average jct
    for _rec in jct_data:
        if (_rec[0] == "crius" or (only_ddl and _rec[0] == "crius-ddl")):
            print(f"[I] Policy: {_rec[0]} | Average JCT: {crius_avg_jct}")
            continue
        # Padding to fairly compare jct
        if len(_rec[1]) > max_finished_job_num:
            print(f"[WARN] Policy {_rec[0]} has more finished jobs ({len(_rec[1])}) than crius ({max_finished_job_num}) because of old data, skip.")
            continue

        # assert len(_rec[1]) <= max_finished_job_num, \
        #     f"Our method should hold the maximum finished " + \
        #     f"job num ({max_finished_job_num}), but less than {_rec[0]} ({len(_rec[1])})"
        _jct_data = _rec[1] + [max_jct] * (max_finished_job_num - len(_rec[1]))

        print(f"[I] Policy: {_rec[0]} | Average JCT: {np.mean(_jct_data)}")

        _ratio = round((np.mean(_jct_data) - crius_avg_jct) / np.mean(_jct_data), 3)
        print("[I] Reduction of Crius on {}: {}".format(_rec[0], _ratio))


##################################
#    Common Uiltity Functions    #
##################################

def _flatten_curve(
    data: Sequence[float], 
    frame_size: int, 
    style: str = "mean",
) -> List[float]:
    """ Flatten curven by averaging in a data frame. """
    
    _list = list()
    frame_size = (frame_size - 1) // 2
    for _i in range(len(data)):
        l_idx = max(_i - frame_size, 0)
        r_idx = min(_i + frame_size + 1, len(data))
        
        if style == "mean":
            _list.append(
                np.mean(data[l_idx:r_idx])
            )
        elif style == "max":
            _list.append(
                np.max(data[l_idx:r_idx])
            )
    
    return _list
