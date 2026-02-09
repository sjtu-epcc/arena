#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to runtime ploting functions.
"""

import os
import json
from pylab import *
from typing import Sequence
from matplotlib import pyplot as plt

""" 
Policy labels.
"""
policy_labels = {
    'crius': 'Crius',
    'fcfs': 'FCFS',
    'fcfs-r': 'FCFS-RE',
    'gandiva': 'Gandiva',
    'elasticflow-l': 'ElasticFlow-LS',
    'gavel': 'Gavel',
    "sia": "Sia",
}
policies = ['crius', 'fcfs', 'gandiva', 'elasticflow-l', 'gavel', 'sia']

# Availble color
# avail_color = ['#434475', '#87BCBD', '#969BC7']
avail_color = ['#434475', '#4E62AB', '#469EB4', '#87CFA4', '#FEE89A', '#FDB96A', '#F57547', '#9E0166']
avail_color = [ "#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#4E62AB" ]


def _read_json_content(json_path):
    """ Json list should be the format as: [{...}, ] """
    with open(json_path, "r", encoding='utf-8') as f:
        json_content = json.load(fp=f)
    return json_content


def _read_thr_data(work_dir: str):
    """ Read all thr data under the work dir. """
    thr_data = list()
    file_names = os.listdir(work_dir)
    for file_name in file_names:
        if (len(file_name.split('.')) != 2) or (len(file_name.split('.')) == 2 and file_name.split('.')[1] != 'npy'):
            continue
        _policy = file_name.split('_')[0]
        _file_path = work_dir + '/' + file_name
        # Load
        _thr_data = list(np.load(_file_path))
        print(f"[I] Policy: {_policy} | Peak throughput: {np.max(_thr_data)} | " + 
              f"Average throughput: {np.mean(_thr_data)} | Sample num: {len(_thr_data)}")
        
        thr_data.append([_policy, _thr_data])
    return thr_data


def plot_jct(work_dir: str):
    """ Read all jct data under the work dir. """
    jct_threshold = 100
    jct_data = dict()
    max_jct, crius_avg_jct, max_finished_job_num = 0, 0, 0
    for _policy in policies:
        json_path = f"{work_dir}/{_policy}_jct.json"
        if not os.path.exists(json_path):
            continue
        jct_table = _read_json_content(json_path)[0]
        jct_data[_policy] = jct_table
        max_jct = max(np.max([jct_table[_jid][1] for _jid in jct_table]), max_jct)

        if _policy == "crius":
            crius_avg_jct = np.mean([jct_table[_jid][1] for _jid in jct_table if jct_table[_jid][1] > jct_threshold])
            max_finished_job_num = len(list(jct_table.keys()))
    
    # Calculate improvement of average jct
    for _policy in jct_data:
        if _policy != "crius":
            # Padding to fairly compare jct
            assert len(jct_data[_policy]) < max_finished_job_num, \
                f"Our method should hold the maximum finished " + \
                f"job num ({max_finished_job_num}), but less " + \
                f"than {_policy} ({len(jct_data[_policy])})"
            _jct_data = [jct_data[_policy][_jid][1] for _jid in jct_data[_policy] if jct_data[_policy][_jid][1] > jct_threshold]
            _finished_job_num = len(_jct_data)
            _jct_data = _jct_data + [max_jct] * (max_finished_job_num - _finished_job_num)
        else:
            _jct_data = [jct_data[_policy][_jid][1] for _jid in jct_data[_policy] if jct_data[_policy][_jid][1] > jct_threshold]

        _mean = np.mean(_jct_data)
        print(f"[I] Policy: {_policy} | Avergae JCT of finished jobs: {_mean}")
        
        if _policy != "crius":
            _ratio = round(crius_avg_jct / np.mean(_jct_data), 3)
            print("[I] Improvement ratio of Crius on {}: {}".format(_policy, _ratio))


def plot_queuing_time(work_dir: str):
    """ Read all queuing time data under the work dir. """
    queuing_time_data = dict()
    for _policy in policies:
        json_path = f"{work_dir}/{_policy}_queuing_time.json"
        if not os.path.exists(json_path):
            continue
        queuing_time_table = _read_json_content(json_path)[0]
        queuing_time_data[_policy] = queuing_time_table

        if _policy == "crius":
            crius_avg_queuing_time = np.mean([queuing_time_table[_jid] for _jid in queuing_time_table])
    
    for _policy in queuing_time_data:
        _mean = np.mean([queuing_time_data[_policy][_jid] for _jid in queuing_time_data[_policy]])
        print(f"[I] Policy: {_policy} | Avergae queuing time of finished jobs: {_mean}")
    
    # Calculate improvement of average queuing_time
    for _policy in queuing_time_data:
        if _policy == "crius":
            continue
        _queuing_time_data = [queuing_time_data[_policy][_jid] for _jid in queuing_time_data[_policy]]

        _ratio = round(crius_avg_queuing_time / np.mean(_queuing_time_data), 3)
        print("[I] Improvement ratio of Crius on {}: {}".format(_policy, _ratio))


def plot_cluster_thr(work_dir: str):
    """
    Plot the cluster throughput throughout the entire trace.
    """
    # Read thr data
    thr_data = _read_thr_data(work_dir)

    # Canvas
    fig, ax1 = plt.subplots(figsize = (15, 4.5), nrows = 1, dpi = 400)
    subplots_adjust(left=0, bottom=0, top=1.0, right=1, hspace=0.3)

    thr_data = sorted(thr_data, key=lambda x: policies.index(x[0]))
    
    # Plot
    for _idx in range(len(thr_data)):
        _len = len(thr_data[_idx][1])
        _pad_len = 600 - _len
        ax1.plot(thr_data[_idx][1] + [0.0 for i in range(_pad_len)], color=avail_color[_idx], label=policy_labels[thr_data[_idx][0]], linewidth=2.5)
    
    # Legend
    ax1.legend(ncol=1,
               loc='upper right',
               fontsize=16,
               markerscale=3,
               labelspacing=0.1,
               edgecolor='black',
               shadow=False,
               fancybox=False,
               handlelength=0.8,
               handletextpad=0.6,
               columnspacing=0.6,
               borderaxespad=0.5,
            )
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=18)

    # Grid
    ax1.grid(axis='y', color='gray', zorder=0)
    
    # Y axis
    y_ticks = [0, 100, 200, 300, 400, 500]
    # y_labels = ['0', '50', '100', '150', '200', '250', '300', '350']
    # plt.yticks(y_ticks, y_labels, fontsize=19, weight='bold')
    plt.yticks(y_ticks, fontsize=19, weight='bold')
    plt.ylim(0.0, 600.0)
    plt.ylabel("Throughput", fontsize=22, weight='bold')

    # X axis
    x_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # x_labels = ['0', '150']
    # plt.xticks(x_ticks, x_labels, fontsize=19, weight='bold')
    plt.xticks(x_ticks, fontsize=19, weight='bold')
    plt.xlim(0.0, 1000.0)
    plt.xlabel("Inspection Round", fontsize=22, weight='bold')

    # Path
    file_path = "./figures/runtime_cluster_thr.pdf" 
    plt.savefig(file_path, bbox_inches='tight')
    # plt.show()
