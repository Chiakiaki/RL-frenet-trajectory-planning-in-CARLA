#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:23:46 2022

@author: sry
"""
from BDPL import ExternalSampler
class Carla_frenet_RL_external_sampler(ExternalSampler):
    """
    custom sampler using RL_frenet_carla_gym
    """
    def __init__(self,carla_env):
        """
        input:
            - env: the gym_env 
        """
        self.env = carla_env
        
    def sample(self,env):
        """
        env need 
        output:
            - ac_candidates: (n,d), need not do normalization since gym will do that again
            
        input:
            - env (a custom env)
            - (P.S., use obs as input is not enough. it may not contains 
               all the necessary info to compute frenet trajectories)
            
        Not necessary in use. We can see that, env.external_sampler and env.step should both be called, so they are better to be defined in a same class
        """
        ac_candidates,num_ac_candidates = self.env.external_sampler()
        return ac_candidates,num_ac_candidates