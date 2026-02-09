#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Data loader which loads training dataset for DNN models.
------------------------------------
- Data loading with PyTorch in Jax: https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html?highlight=data%20loader#data-loading-with-pytorch
- Available list: ['CIFAR10' (for wide-resnet), ]
"""


import os
import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
import jax
import jax.numpy as jnp


def numpy_collate(batch):
    """
    Copied from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html?highlight=data%20loader#data-loading-with-pytorch
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# class NumpyLoader(torch.utils.data.DataLoader):
#     """
#     A tiny shim to make PyTorch DataLoader work with NumPy arrays.
#     --------------------------------
#     Copied from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html?highlight=data%20loader#data-loading-with-pytorch
#     """
#     def __init__(self, dataset, batch_size=1,
#                     shuffle=False, sampler=None,
#                     batch_sampler=None, num_workers=0,
#                     pin_memory=False, drop_last=False,
#                     timeout=0, worker_init_fn=None):
#         super(self.__class__, self).__init__(dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             batch_sampler=batch_sampler,
#             num_workers=num_workers,
#             collate_fn=numpy_collate,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             timeout=timeout,
#             worker_init_fn=worker_init_fn)


class NumPyTransform(object):
    """
    Transform function into numpy for PyTorch dataset.
    """
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)


class DataLoader:
    def __call__(self, dataset_cfgs):
        # Construct kwargs for data_loader

        if dataset_cfgs['dataset_name'] == 'none':
            return None
        
        # print("[I] Loading PyTorch dataset and constructing torch data loader....")
        
        # kwargs = {'batch_size': dataset_cfgs['batch_size']}
        # if dataset_cfgs['use_cuda']:
        #     kwargs.update({'num_workers': 1,
        #                 'pin_memory': True,
        #                 'shuffle': True},
        #                 )
        # else:
        #     # TODO: Unsupported yet.
        #     raise RuntimeError("Not using cuda, unsupported yet.")
        
        # # Define torch dataset & wrapped data loader
        # dataset = None
        # data_loader = None
        # if dataset_cfgs['dataset_name'] == 'CIFAR10':
        #     need_download = False if os.path.exists('./datasets/CIFAR10/cifar-10-batches-py') else True
        #     msg = "[I] Need to download dataset '{}'.".format(dataset_cfgs['dataset_name']) \
        #                         if need_download else "[I] Dataset '{}' has been downloaded.".format(dataset_cfgs['dataset_name'])
        #     print(msg)

        #     dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/', 
        #                                                  train=True, 
        #                                                  transform=NumPyTransform(), 
        #                                                  download=need_download)

        #     data_loader = NumpyLoader(dataset=dataset, **kwargs, )
        # else:
        #     raise RuntimeError("Unsupported dataset. Please recheck the list of available datasets.")
        
        # print("[I] Dataset '{}' loads sucessfully. Dataset Info: \n".format(dataset_cfgs['dataset_name']))
        # print(dataset, '\n')

        # return data_loader


