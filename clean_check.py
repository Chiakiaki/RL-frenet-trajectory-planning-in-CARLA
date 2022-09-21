#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:25:30 2020

This is to test 
1. the three type of formulation of loss for classification task, 

@author: sry
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import queue

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 
    """
    tmp = input_placeholder
    with tf.variable_scope(scope):
        for i in np.arange(n_layers-1):
            tmp = tf.layers.dense(tmp,size,activation=activation,name = str(i+1))
        tmp = tf.layers.dense(tmp,output_size,activation=output_activation,name = str(n_layers),use_bias=False)
    output_placeholder = tmp
    return output_placeholder



bf = queue.Queue(1000)

size = 1000


class model1(object):
    def __init__(self):
        self.D_s = 5
        self.D_label = 5
        self.s_placeholder_nds = tf.placeholder(tf.float32,shape = (None, self.D_s))#currently 'open-loop' RL
        self.gt_placeholder_nd = tf.placeholder(tf.float32,shape = (None, self.D_label))
        self.old_prob_placeholder_nd = tf.placeholder(tf.float32,shape = (None, self.D_label))
        self.logits_nd = build_mlp(self.s_placeholder_nds,self.D_label,'test',3,64)
        self.samples = tf.random.categorical(self.logits_nd,1)
        self.prob_nd = tf.nn.softmax(self.logits_nd)
        opt = tf.train.GradientDescentOptimizer(0.1)


        self.loss1 = - tf.reduce_sum(self.logits_nd * (self.gt_placeholder_nd - tf.stop_gradient(self.prob_nd)),axis = 1)
        self.loss1 = tf.reduce_mean(self.loss1)
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.gt_placeholder_nd,logits = self.logits_nd))
        self.loss3 = tf.reduce_mean(- tf.reduce_sum(tf.log(self.prob_nd) * (self.gt_placeholder_nd) ,axis = 1))
        self.gd1 = opt.compute_gradients(self.loss1)
        self.gd2 = opt.compute_gradients(self.loss2)
        self.gd3 = opt.compute_gradients(self.loss3)
        
        self.stablizing = - tf.reduce_sum( self.logits_nd* (self.old_prob_placeholder_nd - tf.stop_gradient(self.prob_nd)),axis = 1)
        self.stablizing = tf.reduce_mean(self.stablizing)
        
        self.train_op = tf.train.GradientDescentOptimizer(3e-3).minimize(self.loss1)
        self.train_op2 = tf.train.GradientDescentOptimizer(3e-3).minimize(self.loss1+self.stablizing)



input1 = np.array([1,2,3,4,5],dtype = np.float32)
input1 = np.tile(input1,[20,1])
model = model1()


label = np.array([0,0,0,0,1],dtype = np.float32)
label = np.tile(label,[20,1])

"""
#this part is to validate that, the loss1,loss2,loss3 will yield SAME gradients!!!(Note, only when label sum to 1!!!!!!, otherwise loss3 does not provide correct gradient)
#so, minimize loss1,loss2,loss3 are equivalent. They are just three different coding.

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sp,loss1,loss2,loss3,gd1,gd2,gd3 = sess.run([model.samples,model.loss1,model.loss2,model.loss3,model.gd1,model.gd2,model.gd3],feed_dict = {model.s_placeholder_nds:input1,model.gt_placeholder_nd:label})
    sp = np.squeeze(sp)
    lb = np.zeros([20,model.D_label])
    for i in np.arange(20):
        lb[i,sp[i]] = 0
"""


logits_history = []
prob_history = []
loss_history = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in np.arange(10000):

        if (i % 100 == 0):
            print(i)


        """get samples from model, and treat them as labels (lb)"""
        sp,logits,prob = sess.run([model.samples,model.logits_nd,model.prob_nd],feed_dict = {model.s_placeholder_nds:input1})
        sp = np.squeeze(sp)
        lb = np.zeros([20,model.D_label])
        for j in np.arange(20):
            lb[j,sp[j]] = 1

        """ ============================================================="""
        """comment this part, then it will be supervised learning / reinforcement learning with constant reward 1 or -1. We can see the drifting """
        """uncomment this part, then we can see the effect of positive reward(minimize loss1) and negative reward(minimize -loss1)"""
        bf.put(prob)
        
        if(i < 100):
            continue
        
        prob_old = bf.get()
        """We can see that, when set lb[xxx] = -1 constantly, the prob converge to 1/(num_of_catagories). when 1, it will drift. use buffer can stablize.
        It is correct to say, negative reward promote exploration.
        P.S., seems does not provide better effect than -baseline.
        """
        """ ============================================================="""

        """train model with labels"""
        feed_dict = {model.s_placeholder_nds:input1,model.gt_placeholder_nd:lb,model.old_prob_placeholder_nd:prob_old}
        _,_ = sess.run([model.train_op,model.prob_nd],feed_dict = feed_dict)
        prob_history.append(prob)
        logits_history.append(logits)


prob_history = np.array(prob_history)[:,0,:]
plt.plot(prob_history)

plt.figure()
plt.plot(np.var(prob_history,axis = 1))
print(np.mean(np.var(prob_history,axis = 1)[:-2000]))
#6.732265e-05, 4.398291e-05
        
        
        
        
        
        
        
        
        
        
        