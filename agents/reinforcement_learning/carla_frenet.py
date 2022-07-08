#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:23:46 2022

@author: sry
"""
from BDPL import ExternalSampler
class Carla_frenet_RL_external_sampler(ExternalSampler):
    """
    custom sampler using carla
    """
    def __init__(self,env):
    """
    input:
        - env: the gym_env 
    """
        self.env = env
        
    def sample(self,env):
        env.a