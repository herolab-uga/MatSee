#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 20:18:45 2022

@author: sivakr
"""

import os
import os.path as osp
import yaml

'''
Function to check if directory exists
'''
def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.mkdir(dir)
        else:
            raise Exception('Directory not exist: {}'.format(dir))

'''
Function to get training config
'''
def get_train_config(config_file='config/train.yaml'):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    _check_dir(cfg['data_root']['train'], make_dir=False)
    _check_dir(cfg['data_root']['val'], make_dir=False)

    return cfg

'''
Function to get test config
'''
def get_test_config(config_file='config/test.yaml'):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    _check_dir(['data_root']['val'], make_dir=False)

    return cfg
