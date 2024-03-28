#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:26:35 2022

@author: sivakr
"""

from .matnet import MatNet

class DepthNet(MatNet):
    def __init__(self, cfg):
        cfg['in_channels'] = 3
        super(DepthNet, self).__init__(cfg)