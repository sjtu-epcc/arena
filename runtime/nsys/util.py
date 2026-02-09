#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import pickle
from alpa.util import profile_xla_executable


def read_pkl(pkl_pth: str):
    """ Read .pkl file. """
    _f = open(file=pkl_pth, mode='rb')
    prof_db = pickle.load(file=_f)
    _key = ("default", (1, 2))
    prof_res = prof_db[_key]

    print(prof_res.all_gather_cost_dict)
    print(prof_res.all_reduce_cost_dict)
    print(prof_res.all_to_all_cost_dict)
    print(prof_res.reduce_scatter_cost_dict)
    print(prof_res.available_memory_per_device)
    print(prof_res.dot_cost_dict)
    print(prof_res.op_cost_dict)
    


read_pkl(pkl_pth='./prof_database/prof_database.pkl')

