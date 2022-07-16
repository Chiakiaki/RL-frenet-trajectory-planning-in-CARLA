#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 00:24:23 2022

@author: sry
"""
import numpy as np
import scipy as sc


def Process_batch_for_BDP(rollout):
    """
    E.g. if for observation s0, there are a0,a1 two candidates action,
    then we must align them, which means observation batch should be
    [s0,s0], and action batch should be [a0,a1]. num_a_batch will provide
    information for this
    
    
    Previously, a replaybuffer is implemented. Here, for comparison purpose, we do not use it
    Copy past from replaybuffer implementation
    """
    obs_batch, states, rewards, masks, actions, values, ep_infos, true_reward,a_label,prob,all_a, num_a_batch = rollout
    
    a_label = np.squeeze(a_label)#should be 1-D


    tmp3 = [np.ones([1,i]) for i in num_a_batch]
    grouping2 = sc.linalg.block_diag(*tmp3)
    """like, num_a_batch=[2,1,2]
    -> grouping2 = 
    array([[1., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 1.]])

    this is the mask for the numerical problem solved by softmax
    """

    cumsum = np.cumsum(num_a_batch,dtype = np.int32)
    n = cumsum[-1]
    cumsum0 = np.concatenate([[0],cumsum[:-1]])
    a_label_batch = np.zeros([n],dtype = np.float32)
    a_label_idx = a_label + cumsum0
    a_label_batch[a_label_idx] = 1.
    """like, num_a_batch=[2,1,2], a_label = [1,0,1]
    -> a_label_idx = l_idarray([1, 2, 4])        
    all_obs_batch = []
    for (i,_) in enumerate(num_a_batch):
        all_obs_batch += [obs_batch[i]] * num_a_batch[i]
    -> a_label_batch = array([0, 1, 1, 0, 1], dtype=int32)
    """    

    all_obs_batch = []
    def fun2(num_a_batch,obs_batch,all_obs_batch):
        for (i,_) in enumerate(num_a_batch):
            all_obs_batch += [obs_batch[i]] * num_a_batch[i]
    fun2(num_a_batch,obs_batch,all_obs_batch)
    all_obs_batch = np.asarray(all_obs_batch,dtype=np.float32)
    """like, we need to duplicate obs_batch, so that to have same length
    with all_a_batch

    obs_batch = np.eye(3)
    num_a_batch = [2,1,2]

    -> all_obs_batch = 
    [array([1., 0., 0.]),
     array([1., 0., 0.]),
     array([0., 1., 0.]),
     array([0., 0., 1.]),
     array([0., 0., 1.])]
    """
    
    """same for rew_batch,values"""
    all_rew_batch = []
    fun2(num_a_batch,rewards,all_rew_batch)
    all_rew_batch = np.asarray(all_rew_batch,dtype=np.float32)
    all_value = []
    fun2(num_a_batch,values,all_value)
    all_value = np.asarray(all_value,dtype=np.float32)
    
    """"""
    #masks is not used for mlp
    return all_obs_batch,states,all_rew_batch,masks,actions,all_value,ep_infos,true_reward,a_label,prob,all_a, num_a_batch, a_label_batch, grouping2