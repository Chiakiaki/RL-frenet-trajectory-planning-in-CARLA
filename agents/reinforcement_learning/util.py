#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 00:24:23 2022

@author: sry
"""
import numpy as np
import scipy as sc

import tensorflow as tf
from itertools import zip_longest
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm, conv1d

def mlp_extractor(flat_observations, net_arch, act_fun, not_constructing=''):
    """
    Remarks: copy past from policy. add 'not_constracting' to reduce unnecessary memory cost
    
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if not_constructing != 'pi':
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))
        if not_constructing != 'vf':
            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value

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
    fun2(num_a_batch,rewards,all_rew_batch)#rewards:(T,) all rewards:()
    all_rew_batch = np.asarray(all_rew_batch,dtype=np.float32)
    all_value = []
    fun2(num_a_batch,values,all_value)
    all_value = np.asarray(all_value,dtype=np.float32)
    
    """"""
    #masks is not used for mlp
    return all_obs_batch,states,all_rew_batch,masks,actions,all_value,ep_infos,true_reward,a_label,prob,all_a, num_a_batch, a_label_batch, grouping2



def Process_batch_for_BDP_trpo(rollout):
    """
    #for trpo, remove some not needed arguments
    
    E.g. if for observation s0, there are a0,a1 two candidates action,
    then we must align them, which means observation batch should be
    [s0,s0], and action batch should be [a0,a1]. num_a_batch will provide
    information for this
    
    
    Previously, a replaybuffer is implemented. Here, for comparison purpose, we do not use it
    Copy past from replaybuffer implementation
    """
    obs_batch, rewards, values,a_label,all_a, num_a_batch = rollout
    
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
    fun2(num_a_batch,rewards,all_rew_batch)#rewards:(T,) all rewards:()
    all_rew_batch = np.asarray(all_rew_batch,dtype=np.float32)
    all_value = []
    fun2(num_a_batch,values,all_value)
    all_value = np.asarray(all_value,dtype=np.float32)
    
    """"""
    #masks is not used for mlp
    return all_obs_batch,all_rew_batch,all_value,a_label,all_a, num_a_batch, a_label_batch, grouping2,a_label_idx




class ReplayBuffer_QAQ(object):


    def __init__(self, size):
        """specially designed
        
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.        
        """
        self.size = size

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.num_can_sample = 0


        self.obs      = None
        self.all_a    = None
        self.reward   = None
        self.reward2 = None
        self.done     = None
        self.collision = None
        self.a_label  = None #which traj is the gt
        self.num_a    = None #num of trajs for each state

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size <= self.num_can_sample
    
    def _encode_sample(self, idxes, batch_size, sars):
        
        
        all_a_batch    = np.concatenate([ self.all_a[idx] for idx in idxes],axis = 0)
        if self.has_prob:
            prob_batch     = np.concatenate([ self.prob[idx] for idx in idxes],axis = 0)
        else:#no prob stored, but still lets return sth.
            prob_batch     = np.array([ self.prob[idx] for idx in idxes])#this is nothing
        
        if self.has_value:
            value     = self.value[idxes]
            next_value= self.value[(np.array(idxes) + 1) % self.num_in_buffer]
        else:
            value     = None
            next_value= None
            
        rew_batch      = self.reward[idxes]
        rew2_batch     = self.reward2[idxes]        
        obs_batch      = self.obs[idxes]
        next_obs_batch = self.obs[(np.array(idxes) + 1) % self.num_in_buffer]
        a_tmp        = self.a_label[idxes]
        num_a_batch = self.num_a[idxes]
        done_batch = self.done[idxes]

        def fun1(num_a_batch):
            tmd1 = []
            tmd2 = []
            tmp2 = np.cumsum(num_a_batch,dtype = np.int32)
            tmp3 = np.concatenate([[0],tmp2[:-1]])
            for (i,_) in enumerate(num_a_batch):
                tmd1 += [tmp2[i]] * num_a_batch[i]
                tmd2 += [tmp3[i]] * num_a_batch[i]
            n = np.shape(tmd1)[0]
            ret1 = (tmd1,np.arange(n,dtype = np.int32))  
            ret2 = (tmd2,np.arange(n,dtype = np.int32)) 
            return ret1,ret2,tmp2
            
        
        
        tmp1,tmp2,cumsum = fun1(num_a_batch)


        """like, num_a_batch=[2,1,2]
        fun1(num_a_batch) -> 
        tmp1 = ([2, 2, 3, 5, 5], array([0, 1, 2, 3, 4], dtype=int32)
        tmp2 = ([0, 0, 2, 3, 3], array([0, 1, 2, 3, 4], dtype=int32))
        
        -> grouping = 
        [ , ,  ,  ,  ]
        [1,1,-1,  ,  ]
        [ , , 1,-1,-1]
        [ , ,  ,  ,  ]
        [ , ,  , 1, 1]
        """
        n = np.shape(tmp1[0])[0]
        grouping = np.zeros([n+1,n],dtype = np.int32)
        grouping[tmp1] = 1
        grouping[tmp2] = -1
        grouping = grouping[1:,:]

        """like, num_a_batch=[2,1,2]
        -> grouping2 = 
        array([[1., 1., 0., 0., 0.],
               [0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 1.]])

        this is the mask for the numerical problem solved by softmax
        """
        tmp3 = [np.ones([1,i]) for i in num_a_batch]
        grouping2 = sc.linalg.block_diag(*tmp3)

        a_label_batch = np.zeros([n],dtype = np.int32)
        cumsum0 = np.concatenate([[0],cumsum[:-1]])
        a_label_idx = a_tmp + cumsum0
        a_label_batch[a_label_idx] = 1
        act_batch = all_a_batch[a_label_idx]
        """like, num_a_batch=[2,1,2], a_tmp = [1,0,1]
        -> a_label_idx = l_idarray([1, 2, 4])        all_obs_batch = []
        for (i,_) in enumerate(num_a_batch):
            all_obs_batch += [obs_batch[i]] * num_a_batch[i]
        -> a_label_batch = array([0, 1, 1, 0, 1], dtype=int32)
        """

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
        all_obs_batch = []
        def fun2(num_a_batch,obs_batch,all_obs_batch):
            for (i,_) in enumerate(num_a_batch):
                all_obs_batch += [obs_batch[i]] * num_a_batch[i]
        fun2(num_a_batch,obs_batch,all_obs_batch)

        """same for rew_batch"""
        all_rew_batch = []
        fun2(num_a_batch,rew_batch,all_rew_batch)
        all_rew2_batch = []
        fun2(num_a_batch,rew2_batch,all_rew2_batch)
#        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        next_idxes = (np.array(idxes) + 1) % self.num_in_buffer
        if sars == True:
            next_state_things = self._encode_sample(next_idxes,batch_size,sars = False)
            return all_obs_batch, all_a_batch, prob_batch, all_rew_batch, all_rew2_batch, grouping, a_label_batch, grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch,num_a_batch,value,next_value,next_state_things

        return all_obs_batch, all_a_batch, prob_batch, all_rew_batch, all_rew2_batch, grouping, a_label_batch, grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch,num_a_batch,value,next_value


    def sample(self, batch_size, sars = False):
        #by default, return s-a-r

        assert self.can_sample(batch_size)
        
        idxes = sample_n_unique(lambda: random.randint(0, self.num_can_sample - 1), batch_size)
        idxes = idxes + self.shift_of_idxes[idxes]#we don't want to sample out a 'done but not collision' state 
        
#        if sars == True:
#            idxes = sample_n_unique(lambda: random.randint(0, self.num_can_sample - 1), batch_size)
#            idxes = idxes + self.shift_of_idxes[idxes]
#        else:
#            idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 1), batch_size)

        return self._encode_sample(idxes,batch_size,sars = sars)


    def store_frames2(self,rollout):
        """
        wrapper to use this replay buffer
        """
        obs_batch, states, rewards, masks, actions, values, ep_infos, true_reward,a_label,prob,all_a, num_a_batch = rollout
        raise NotImplementedError

    def store_frames(self, s_episode, all_a_episode, prob_episode, r_episode, r2_episode, a_label_episode, num_traj_list, done,collision = None,value_episode = None):
        """
        store multiple frames, mostly for a batch or episode

        Parameters
        ----------
        s_episode - list
        all_a_episode - list of array
        r_episode - list
        r2_episode - list
        a_label_episode - list (not one-hot)
        num_traj_list - list
        done - list. If it is a dead state. 1 == dead
        value_episode - list
        """
        
        
        
        if collision is None:
            collision = done
        
        n = np.shape(s_episode)[0]        
        n_a = np.sum(num_traj_list)

        st = self.next_idx
        ed = self.next_idx+n
        
        if n == 0:
            return st
        
        if prob_episode == None:
            self.has_prob = False
            prob_episode = list(np.ones(n))
        else:
            self.has_prob = True
            prob_episode = list(prob_episode)
            
        if value_episode is not None:
            self.has_value = True
        else:
            self.has_value = False

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(s_episode[0].shape), dtype=np.float32)
            self.all_a    = []#dynamic vector
            self.prob     = []#dynamic vector
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.reward2  = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.int32)
            self.collision     = np.empty([self.size],                     dtype=np.int32)
            self.a_label  = np.empty([self.size],                     dtype=np.int32)
            self.num_a  = np.empty([self.size],                     dtype=np.int32)
            self.value    = np.empty([self.size],                     dtype=np.float32)

        if ed > self.size:
            part2 = ed - self.size            
            part1 = self.size - st#always >= 1
            self.obs[st:st+part1] = s_episode[:part1]
            self.obs[:part2] = s_episode[part1:]
            if self.num_in_buffer < self.size:
                self.all_a += all_a_episode[:part1]
                self.prob += prob_episode[:part1]
            else:
                self.all_a[-part1:] = all_a_episode[:part1]
                self.prob[-part1:] = prob_episode[:part1]
            self.all_a[:part2] = all_a_episode[part1:]
            self.prob[:part2] = prob_episode[part1:]
            self.reward[st:st+part1] = r_episode[:part1]
            self.reward[:part2] = r_episode[part1:]
            self.reward2[st:st+part1] = r2_episode[:part1]
            self.reward2[:part2] = r2_episode[part1:]
            self.a_label[st:st+part1] = a_label_episode[:part1]
            self.a_label[:part2] = a_label_episode[part1:]
            self.num_a[st:st+part1] = num_traj_list[:part1]
            self.num_a[:part2] = num_traj_list[part1:]
            self.done[st:st+part1] = done[:part1]
            self.done[:part2] = done[part1:]
            self.collision[st:st+part1] = collision[:part1]
            self.collision[:part2] = collision[part1:]
            
            if self.has_value:
                self.value[st:st+part1] = value_episode[:part1]
                self.value[:part2] = value_episode[part1:]                


        else:
            self.obs[self.next_idx:self.next_idx+n] = s_episode
            self.all_a[self.next_idx:self.next_idx+n] = all_a_episode
            self.prob[self.next_idx:self.next_idx+n] = prob_episode
            self.reward[self.next_idx:self.next_idx+n] = r_episode
            self.reward2[self.next_idx:self.next_idx+n] = r2_episode
            self.a_label[self.next_idx:self.next_idx+n] = a_label_episode
            self.num_a[self.next_idx:self.next_idx+n] = num_traj_list
            self.done[self.next_idx:self.next_idx+n] = done
            self.collision[self.next_idx:self.next_idx+n] = collision
            
            if self.has_value:
                self.value[self.next_idx:self.next_idx+n] = r_episode
            
        self.next_idx = (self.next_idx + n) % self.size

        self.num_in_buffer = min(self.size, self.num_in_buffer + n)
        self.done_not_collision = np.where(self.collision[:self.num_in_buffer] != self.done[:self.num_in_buffer])[0]

        self.num_can_sample = self.num_in_buffer - len(self.done_not_collision)#for special treatment for 'done' but not 'collision'
        
        tmp = np.zeros(self.num_in_buffer,dtype = np.int32)
#        tmp[self.done_not_collision] = 1#this is wrong code!!!!!!!!!!!!
#        self.shift_of_idxes = np.cumsum(tmp)#this is wrong code!!!!!!!!!!!!
#       below is the right one
        tmp2 = self.done_not_collision - np.arange(len(self.done_not_collision))
        tmp[tmp2] += 1
        self.shift_of_idxes = np.cumsum(tmp)
        
        
        return st