#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:16:45 2022

@author: sry
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from gym import spaces

from stable_baselines.a2c.utils import linear
from stable_baselines.common.distributions import ProbabilityDistributionType,ProbabilityDistribution,CategoricalProbabilityDistribution


class BoltzmannDistributionV1(CategoricalProbabilityDistribution):
    """
    Remarks: Modified from catagorical distribution. Not using the original catagorical distribution is because
    its 'logits' is of shape (n,n_cat), which means, the number catagories need to be the same across batch. Boltzmann Distribution has not defined number of catagories.
    So, to support batch computation, I have to do this, unless we does not use batch.
    """
    def __init__(self, goodness_n, grouping_mn):
        """
        grouping_mn: (m,n) tensor
        goodness_n: (n,) tensor
        """
        """
        # ********* Here is an example for what we are doing, can run directly********* #
        # goodness_n is the "goodness" or "fitness" of 
        # state - action pair: (s0,a0),(s0,a1),(s1,a2),(s2,a3),(s2,a4),
        # and the probability of a at state s should be porpotional to exp(goodness)
        # Lets say, their values are,
        goodness_n = np.array([-6001.,-6002.,-6003.,-6004.,-6005])
        goodness_n = np.array([-1.,-2.,-3.,-4.,-5])
        goodness_n = np.array([1.,2.,3.,4.,5])
        goodness_n = np.array([6001.,6002.,6003.,6004.,6005])
        goodness_n = tf.constant(goodness_n)
        #
        # Then grouping_mn is like follow (watch the differences for s0,s1,s2) 
        grouping_mn = np.array([[1., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.],
                                [0., 0., 0., 1., 1.]])
        grouping_mn = tf.constant(grouping_mn)
        
        # Then, to calculate the probability, we will have:

        
        inf_mask = 1/grouping_mn - grouping_mn
        g_est_mn = goodness_n * grouping_mn#g_est_mn, means it measures how good of each state_action pair
        g_est_mn = g_est_mn - inf_mask #hint: exp(-inf) is 0
        prob_mn = tf.nn.softmax(g_est_mn)
        prob_n = tf.reduce_sum(prob_mn,axis = 0)
        prob_n = tf.stop_gradient(prob_n)
        
        with tf.Session() as sess:
            prob_mn,prob_n,g_est_mn = sess.run([prob_mn,prob_n,g_est_mn])
        """
        
        inf_mask = 1/grouping_mn - grouping_mn
        inf_mask = tf.stop_gradient(inf_mask)
        logits = goodness_n * grouping_mn#g_est_mn, means it measures how good of each state_action pair
        logits = logits - inf_mask #hint: exp(-inf) is 0
        #logits: (m,n)
        self.grouping_mn = tf.stop_gradient(grouping_mn)
        
        super(BoltzmannDistributionV1, self).__init__(logits)#this will do self.logits = logits




    
    def neglogp(self, x, one_hot_input_flag):
        # original code saying "Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives..."
        # remarks: we cannot use the code in CategoricalProbabilityDistribution since
        #           the tf.one_hot does not support (None,) shape. It also 
        #           does not support -inf in logits
        # self.logits has (None,None) shape
        
        # on run, logits:(m,n), not (n,num_catagory), and most is zero
        # x: (m),should be a_label_m in case of action
        # very careful here!!!!!!!!!!! in batch computation, when x is often the
        # action, it should be processed!!!
        # e.g. action = [1,0,1], num_a_n = [3,2,3], then the real x input should
        # be [1,3,6]!!!
        #
        # one_hot_input_flag is to force check this functhon whenther call it
        """
        let me explain more about original code saying "Note: we can't use 
        sparse_softmax_cross_entropy_with_logits because he implementation does not 
        allow second-order derivatives..."
        
        Gradient is a tensor. so, calculate gradient of gradient is programmable, and is
        actually how the fisher vector product is calculated. However, did not observe
        error when using it like below:
        
        # 2nd gradient with sparse_softmax_cross_entropy_with_logits, will yield error in tensorflow 1.15
        import tensorflow as tf
        import numpy as np
        labels = tf.constant([0])
        logits1 = tf.get_variable("qaq",shape=(1,2),initializer=tf.constant_initializer([1,2]))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits1,
                labels=labels)
        var_list = tf.trainable_variables()
        grad = tf.gradients(loss,var_list)
        sum1 = tf.reduce_sum(grad)
        grad2 = tf.gradients(sum1,var_list)
        with Session() as sess:
            g1,g2 = sess.run([grad,grad2])
        # ========================
        
        The good news is that, in bdp, we only need 2nd order derivatives of kl.
        We does not need 2nd order of sparse_softmax_cross_entropy_with_logits.
        """

        
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=x)#this will output 'nan' when computing grad of grad

        
        """
        one_hot_actions = tf.one_hot(x, 24)
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))
        """
        
        
        
        
        
    def logp(self, x, one_hot_input_flag):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x, one_hot_input_flag)
    

    def kl(self, other):
        """
        Here we need to fix a issue when there is -inf in logits
        If there is 0 probability in certain catagory, the previous kl will yield error
        Note see https://math.stackexchange.com/questions/1228408/kullback-leibler-divergence-when-the-q-distribution-has-zero-values
        for kl with 0 in probability
        (Which happen to be the case in Boltzmann Policy)
        """
        
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        
        #z_0 and z_1  (the partition) is correct when has -inf
        #a_0 - a_1 will has nan, the correct value is just 0
        exp_a_0_ = exp_a_0 + 1 - self.grouping_mn
        exp_a_1_ = exp_a_1 + 1 - self.grouping_mn
        a_0_ = tf.log(exp_a_0_)
        a_1_ = tf.log(exp_a_1_)
        
        return tf.reduce_sum(p_0 * (a_0_ - tf.log(z_0) - a_1_ + tf.log(z_1)), axis=-1)

        
    def entropy(self):
        """
        fix the issue for -inf in logits (0 probability in catagory)
        """
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        
        exp_a_0_ = exp_a_0 + 1 - self.grouping_mn
        a_0_ = tf.log(exp_a_0_)
        
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0_), axis=-1)
    
    
    """
    # ********* Here is an example to test all above, can run directly********* #
    # goodness_n is the "goodness" or "fitness" of 
    # state - action pair: (s0,a0),(s0,a1),(s1,a2),(s2,a3),(s2,a4),
    # and the probability of a at state s should be porpotional to exp(goodness)
    # Lets say, their values are,
    import tensorflow as tf
    import numpy as np
    goodness_n = np.array([-6001.,-6002.,-6003.,-6004.,-6005])
    goodness_n = np.array([-1.,-2.,-3.,-4.,-5])
    # goodness_n = np.array([1.,2.,3.,4.,5])
    goodness_n2 = np.array([6001.,6002.,6003.,6004.,6005])
    goodness_n = tf.constant(goodness_n)
    #
    # Then grouping_mn is like follow (watch the differences for s0,s1,s2) 
    grouping_mn = np.array([[1., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 1.]])
    grouping_mn = tf.constant(grouping_mn)
    
    # Then, to calculate the probability, we will have:

    pd = BoltzmannDistributionV1(goodness_n,grouping_mn)
    pd2 = BoltzmannDistributionV1(goodness_n2,grouping_mn)
    
    # kl
    a_0 = pd.logits - tf.reduce_max(pd.logits, axis=-1, keepdims=True)
    a_1 = pd2.logits - tf.reduce_max(pd2.logits, axis=-1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    exp_a_1 = tf.exp(a_1)
    z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
    z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
    p_0 = exp_a_0 / z_0
    logz0 = tf.log(z_0)
    exp_a_0_ = exp_a_0 + 1 - grouping_mn
    exp_a_1_ = exp_a_1 + 1 - grouping_mn
    a_0_ = tf.log(exp_a_0_)
    a_1_ = tf.log(exp_a_1_)
    a0_minus_a1 = a_0_ - a_1_
    kl = pd.kl(pd2)
    
    # entropy
    ent = pd.entropy()
    
    # logp
    a_label = tf.constant([1,2,3])
    logp = pd.logp(a_label,None)
    

    
    with tf.Session() as sess:
        logits,logits2,a0,a1,z0,p0,logz0,a0_minus_a1,kl_value,ent = sess.run([pd.logits,pd2.logits,a_0,a_1,z_0,p_0,logz0,a0_minus_a1,kl,ent])
        
    with tf.Session() as sess:
        logp = sess.run(logp)
        

        
        
    """