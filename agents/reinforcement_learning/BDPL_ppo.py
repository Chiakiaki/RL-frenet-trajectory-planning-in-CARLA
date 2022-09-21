#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:55:16 2022

@author: sry

modified from openai stable baseline a2c

"""
import time

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import discount_with_dones, Scheduler, mse, total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean

"""Added imported for BDPL"""
from abc import ABC, abstractmethod
from gym.spaces import Discrete
from stable_baselines.common.policies import BasePolicy,nature_cnn,sequence_1d_cnn,mlp_extractor
from stable_baselines.common.input import observation_input
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm, conv1d
import warnings
#from carla_frenet_RL_external_sampler import Carla_frenet_RL_external_sampler
from util import Process_batch_for_BDP,ReplayBuffer_QAQ
"""End"""


class BoltzmannDistribution(object):
    """
    Tried to rewrite Boltzmann policy in openai Gym's style. However, only limited number
    of method is supported.
    
    Note, the input for __init__, should be tensor, not an actual value. The distribution
    should not be fixed
    """
    def __init__(self):
        super(BoltzmannDistribution, self).__init__()
        
    def sample(self,acs_candidates=None):
        assert acs_candidates is not None and type(acs_candidates) is tf.Tensor, "actions_candidates should not be none, and should be a tensor."
        

class BoltzmannDistributionPolicy(object):
    """
    Tried to rewrite Boltzmann policy in openai Gym's style.

    add 'grouping' tensor
    
    ***Main changes are inside the "_setup_init" method***
    """
    recurrent = False
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=True):
        #now copy and past from stable baselines
        
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)#(n_batch,[action_space])
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            
            """BDPL changes"""
            """Notes: Boltzmann Distribution does not has an analytics
            formulation to calculate the brobability of each sample. It does not has parameters 
            (e.g., mean and var are parameters for Gaussian). Whereas, it compute how 'good' or 
            'fit' of an sample COMPARED WITH OTHERS. So, here we will add candidate actions as 
            input, and another 'grouping' vector as input to deal with batch computation: we 
            need to know, inside a batch, which inputs are inside which comparison group.
            """
            assert add_action_ph==True, "Error: must create action placeholder for Boltzmann Policy"
            if add_action_ph:
                #the observation_input will return a tensor that is casted and normalized
                #lets say we process acs like obs
                self._action_ph, self._processed_acs = observation_input(ac_space, n_batch, scale=scale,name="action_ph")
            

            self.grouping_ph_mn = tf.placeholder(dtype=tf.float32, shape=(None, n_batch))#note, this placeholder could be of shape (None,none)
            self.prob_ph_n = tf.placeholder(dtype=tf.float32, shape=(n_batch))#input for probability calculated by Boltzmann Distribution
            
            """End of changes"""
            
        
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space
        
        self.sample_a_idx = None
        """Copy past with some modification"""
        self._policy_latent = None
        self._value_fn = None
#        self._action = None
        self._deterministic_action = None
        #self._pdtype = make_proba_dist_type(ac_space)  ??? not used
        #self._proba_distribution = None  ??? not defined
        
        """End of changes"""
    
    def _setup_init(self):
        """
        Gym's style, sets up the Boltzmann Distribution, build output computational graph.
        They define here:
            self._action
            self._deterministic_action
            self._neglogp
            self._policy_proba
            self._value_flat
        
        We define here:
            self.prob_n
            self.goodness_n
            
        """
        """BDPL changes Here"""
        with tf.variable_scope("output", reuse=True):
            assert self._policy_latent is not None
            _policy_latent = self._policy_latent[:,0]#(n,1)->(n,)
            """
            # ********* Here is an example for what we are doing, can run directly********* #
            # _policy_latent is the "goodness" or "fitness" of 
            # state - action pair: (s0,a0),(s0,a1),(s1,a2),(s2,a3),(s2,a4),
            # and the probability of a at state s should be porpotional to exp(goodness)
            # Lets say, their values are,
            _policy_latent = np.array([-6001.,-6002.,-6003.,-6004.,-6005])
            _policy_latent = np.array([-1.,-2.,-3.,-4.,-5])
            _policy_latent = np.array([1.,2.,3.,4.,5])
            _policy_latent = np.array([6001.,6002.,6003.,6004.,6005])
            _policy_latent = tf.constant(_policy_latent)
            #
            # Then grouping_ph_mn is like follow (watch the differences for s0,s1,s2) 
            grouping_ph_mn = np.array([[1., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 1.]])
            grouping_ph_mn = tf.constant(grouping_ph_mn)
            
            # Then, to calculate the probability, we will have:
            inf_mask = 1/grouping_ph_mn - grouping_ph_mn
            g_est_mn = _policy_latent * grouping_ph_mn
            g_est_mn = g_est_mn - inf_mask # Hint: exp(-inf) is 0
            prob_mn = tf.nn.softmax(g_est_mn)
            prob_n2 = tf.reduce_sum(prob_mn,axis = 0)
            with tf.Session() as sess:
                prob_mn,prob_n2,g_est_mn = sess.run([prob_mn,prob_n2,g_est_mn])
            """
            inf_mask = 1/self.grouping_ph_mn - self.grouping_ph_mn
            g_est_mn = _policy_latent * self.grouping_ph_mn#g_est_mn, means it measures how good of each state_action pair
            g_est_mn = g_est_mn - inf_mask #hint: exp(-inf) is 0
            prob_mn = tf.nn.softmax(g_est_mn)
            prob_n = tf.reduce_sum(prob_mn,axis = 0)
            prob_n = tf.stop_gradient(prob_n)
            
            self.prob_n = prob_n
            self.goodness_n = _policy_latent
            
            #on-policy sampling
            self.sample_a_idx = tf.random.categorical(tf.expand_dims(self.goodness_n,axis = 0),1)#(1,1)
            """End"""

    
    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs
    
    """BDPL changes"""
    @property
    def processed_acs(self):
        """like processed_obs, see processed_obs for more information"""
        return self._processed_acs
    """End"""

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))


class FeedForwardBoltzmannDistributionPolicy(BoltzmannDistributionPolicy):
    #this is an actor-critic policy
    #copy and past from FeedForwardPolicy, but change to Boltzmann Distribution
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardBoltzmannDistributionPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]
        

        """"""
        """Now start to chage
        Normally, the openai's mlp_extractor will create both policy and
        value network. However, here we have different input for policy 
        and value, so we need to define them separately
        
        Note, we cannot share variable between policy and value function,
        since they have different input dimension"""
        """"""
        #firstly, add action into model input
        with tf.variable_scope("policy_model", reuse=reuse):
            if feature_extraction == "cnn":
                raise NotImplementedError
#                pi_latent = cnn_extractor(processed_input, **kwargs)
            else:
                processed_input = tf.concat([tf.layers.flatten(self.processed_obs),self.processed_acs], axis=-1)#(N,[D_obs])#note!!!not necessarily (N,D),since ob.space can be, e.g. (N,5,9), instead of (N,45,)
                pi_latent,_ = mlp_extractor(processed_input, net_arch, act_fun)
            self._policy_latent = linear(pi_latent, 'pi', 1)#policy model, a tensor


            
        with tf.variable_scope("critic", reuse=reuse):
            if feature_extraction == "cnn":
                vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                vf_latent,_ = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)
            
            self._value_fn = linear(vf_latent, 'vf', 1)#(batch_size,1)
            #self.q_values = ??? ??NotImplemented              
            
        """
        In stable baseline, they firstly define and initialize 'probability' class here, 
        then call the _setup_init() with 'probability', which
        will define following tensors:
            self._action
            self._deterministic_action
            self._neglogp
            self._policy_proba
            self._value_flat
        """
        self._setup_init()#this will need self._policy_latent defined
        
        self._value_flat = self._value_fn[:,0]

        
    
    def step(self, obs, ac_candidates, state=None, mask=None, deterministic=False):
        #feed action_ph,obs_ph,
        assert np.shape(obs)[0] == 1, "batch computation is only used for training, does not support batch here"
        if deterministic:
            raise NotImplementedError
        else:
            #note, after the obs and ac_candidates are feeded in, if 'scale == true', the values of them will be scaled by space.high, and space.low. scale == true is set if using cnn_extractor and else false by default.
            #tile obs to the same shape of ac_candidates
            tile_args = len(obs.shape)*[1]
            num_traj = np.shape(ac_candidates)[0]
            tile_args[0] = num_traj
            all_obs = np.tile(obs,tile_args)
#            grouping_ph_mn
#            td_map
            grouping = np.array([[1]*num_traj])
            
            sample_a_idx, value, prob_n = self.sess.run([self.sample_a_idx,self.value_flat,self.prob_n], feed_dict={self.obs_ph:all_obs, self.action_ph:ac_candidates,self.grouping_ph_mn:grouping})
            return sample_a_idx, value, self.initial_state, prob_n
            
    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn
    
    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat
            
    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class BDPL(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=0, pg_lr_multiplyier=1,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, model_dir=None):

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.model_dir = model_dir

        self.learning_rate_ph = None
        self.n_batch = None
        self.m_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
#        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        self.pg_lr_multiplyier = pg_lr_multiplyier

        super(BDPL, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        
        
        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return BDPLRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):
        raise NotImplementedError # only copy past from openai baseline a2c what did this part do??????????????????????
        policy = self.train_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.actions_ph, policy.policy
        return policy.obs_ph, self.actions_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, BoltzmannDistributionPolicy), "Error: the input policy for the BDPL model must be an " \
                                                                "instance of BoltzmannDistributionPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.train_global_step = tf.get_variable('BDPL_global_step',[], initializer = tf.constant_initializer(0), trainable=False)
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.n_batch = None#we will not know, is the actual batch size for computation
                self.m_batch = self.n_envs * self.n_steps#the real conventional 'batch'

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    raise NotImplementedError
                    
                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                                             n_batch_step, reuse=False, **self.policy_kwargs)                    

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    #actions_ph is defined in policy model
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    """extra definition for BDPL"""
                    self.gt_placeholder_n = tf.placeholder(dtype=tf.float32, shape = (None))
                    
                    """
                    # The probability of (s,a) is already calculated once when do on-policy sampling
                    # here, we can calculate it again, or , define a placeholder and use the already 
                    # calculated value
                    # See below: prob_n is the recalculated value, prob_n2 is the already calculated one
                    """
                    self.prob_placeholder_n = tf.placeholder(dtype=tf.float32,shape = (None))
                    prob_n = train_model.prob_n
                    prob_n2 = self.prob_placeholder_n
                    self.loss1 = - tf.reduce_mean(train_model.goodness_n * (self.gt_placeholder_n - prob_n2) * self.advs_ph)
                    self.loss2 = - tf.reduce_mean(train_model.goodness_n * (prob_n2 - prob_n) * 2.)#debug and testing
                    self.pg_loss = self.loss1 + self.loss2#loss 2 is for debug
                    self.debug_info = self.gt_placeholder_n - prob_n2
                    
                    self.vf_loss = mse(tf.squeeze(train_model.value_flat), self.rewards_ph)
                    loss = self.pg_loss * self.pg_lr_multiplyier + self.vf_loss

#                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('loss', loss)
                    tf.summary.scalar('debug_loss2', self.loss2)
                    tf.summary.scalar('prob_n2:0', prob_n2[0])
                    tf.summary.scalar('prob_n2:1', prob_n2[1])
                    tf.summary.scalar('prob_n2:2', prob_n2[2])

                    self.params = tf_util.get_trainable_vars("policy_model") + tf_util.get_trainable_vars("critic") 
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                    epsilon=self.epsilon)
#                trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
#                                    epsilon=self.epsilon)
    
                self.apply_backprop = trainer.apply_gradients(grads,global_step = self.train_global_step)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
#                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

                #TODO: os.system("cp ...)

                self.summary = tf.summary.merge_all()

    def _train_step(self, obs, states, rewards, masks, actions, values, update, grouping_mn = None,gt_n = None,prob_n = None, writer=None):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        
        Note for using BDPL: the state and action though is one batch, they are of different group
        
        """
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.train_model.action_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        if states is not None:
            raise NotImplementedError#rnn not implemnted
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks
            
        """Now add 'grouping_ph_mn' and 'gt_placeholder_n' in for BDPL"""
        assert grouping_mn is not None
        assert gt_n is not None
        td_map[self.train_model.grouping_ph_mn] = grouping_mn
        td_map[self.gt_placeholder_n] = gt_n
        td_map[self.prob_placeholder_n] = prob_n
        

        
        #modifined below: n_batch is not the previous meanings now. It has unknow value
        #e.g. (s0,a0),(s0,a1),(s1,a0),(s2,a0),(s2,a1) has n_batch 5, but actuallly they are only 3 timesteps
        #policy entropy is deleted
        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, global_step, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.train_global_step,self.apply_backprop],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * self.m_batch))
            else:
                summary, policy_loss, value_loss, global_step, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.train_global_step,self.apply_backprop], td_map)
            writer.add_summary(summary, update * self.m_batch)

        else:
            policy_loss, value_loss, global_step, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.train_global_step, self.apply_backprop], td_map)

        return policy_loss, value_loss, None

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="BDPL",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            t_start = time.time()
            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.m_batch + 1):

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # unpack,check shape here
                # (15,5,9),
                obs, states, rewards, masks, _, values, ep_infos, true_reward,a_label,prob,all_a,num_traj,gt_n,grouping_mn = Process_batch_for_BDP(rollout)

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break
                
                # TODO: grouping_mn = None,gt_n = None,action->all_a_action
                # processing for bdpl
                
                
                
                
                self.ep_info_buf.extend(ep_infos)
                _, value_loss, _ = self._train_step(obs, states, rewards, masks, all_a, values, self.num_timesteps // self.m_batch, grouping_mn = grouping_mn,gt_n = gt_n, prob_n = prob, writer = writer)#value is for computing adv
                n_seconds = time.time() - t_start
                fps = int((update * self.m_batch) / n_seconds)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)


                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
#                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.dump_tabular()

                # Save time model every 5000 step
                if self.num_timesteps % 5000 == 0:
                    self.save(self.model_dir + "/step_{}".format(self.num_timesteps))


        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
        
    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class ExternalSampler(ABC):
    """
    External sampler for boltzmann distribution policy learning.
    Output: candidate actions
    """
    def __init__(self,**kwargs):
        pass
    
    @abstractmethod
    def sample(self):
        raise NotImplementedError
    
class ExternalSampler_GymDiscreteAction(ExternalSampler):
    """
    If action space is discrete, we will simply return all action
    """
    def __init__(self,ac_space,**kwargs):
        super(ExternalSampler_GymDiscreteAction, self).__init__()
        self.ac_space = ac_space
        assert isinstance(self.ac_space,Discrete)
        
    def sample(self):
        return np.arange(self.ac_space)



    
    

class BDPLRunner_ppo(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(BDPLRunner_ppo, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma
        
        #replay_buffer
        self.replay_buffer = ReplayBuffer_QAQ(size = 500)

    def _run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        assert self.n_envs == 1
        """
        Some explaination to n_env > 1:
            Normally, when env.step is called, the returned obs is of shape (1,[D_obs])
            OpenAI has vec_env, which is a wraper to wrap up multiple envs, and by called
            '.step' method of such vec_env, the returnd obs will be of shape [n_env,[D_obs]]
            However, I think the carla gym does not has such support (maybe it can?), and 
            we just use one env at a time.
            
            P.S.: Doing env.step(actions) when actions is not a list or is just (D_acs), openai's
            vec_env will process it. Whether can model.step(obs) do obs of shape (n_env,D_acs) is
            depend on model (policies), but normally, our model.step to support batch computation,
            so normally we does not need special treatment to model.step, but for BDPL, since
            we have ac_candidates as input, things will be different. 
            
        In short, n_env > 1 is not supported here, unless following part is improved:
            The above env.external_sampler() is not of OpenAI's Gym, it only work as n_env == 1.
            Even wifh n_env == 1, self.obs will still be of shape (m,[D_obs]), m is number of candidates
            
        """

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        #add 4 more list for BDPL
        a_label_episode, prob_episode, all_a_episode, num_a_list = [],[],[],[]
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            # for BDP
            # the env is always wrapped as vecEnv, so we refer to our defined function with env.envs[0]
            ac_candidates,n_ac_candidates = self.env.envs[0].external_sampler()#ac_candidates:(n,d)
            assert len(np.shape(ac_candidates)) == 2, "ac_candidates' dims should be 2"
            actions_idx, values, states, prob_n = self.model.step(self.obs, ac_candidates, self.states, self.dones)# note: action_idx is of shape (1,1),not scalar
            #value's shape is in accordance with processed_obs. here is (sum of n_ac_candidates accross n_env),so when n_env is 1, it will be (n_ac_candidates)
            assert self.n_envs == 1
            values = values[[0]]#only work when self.n_env == 1, as we have said , the returned values is actually all_values

            
            
            # actions_idx:(1,1),
            actions_idx = actions_idx[0]#now (1,)
            # self.obs will be feed into placeholder of (n_batch,[ob_space]) ac_candidates
            # will be of the same shape. However!!! n_batch should be T*n_env*n. Ob_space should be tiled
            action = ac_candidates[actions_idx,:]#(1,d),since actions_idx is (1,)
            d_action = np.shape(action)[1:]

            mb_obs.append(np.copy(self.obs))#T x (1,[d_s]),[d_s] is may be 2-d, like (5,9)
            mb_actions.append(action)#T x (1,d)#1 is because n_env is 1
            mb_values.append(values)#
            mb_dones.append(self.dones)#T x (1), so it will be (T,1), not (T)
            
            a_label_episode.append(actions_idx)#(T,1)
            prob_episode.append(prob_n)#(T,n)
            all_a_episode.append(ac_candidates)#(T,?)
            num_a_list.append(n_ac_candidates)#(T,n,d)
            
            
#            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
            #the environment will find action by ac_candidates[actions_idx,:]
            #action_idx is of shape (n_env,)
            obs, rewards, dones, infos = self.env.step(actions_idx)#(1,5,9),(1,),(1,),_
            # self.env.render()

            self.model.num_timesteps += self.n_envs
            
            # TODO: early stop, this will end the training
            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 12

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape([-1] + list(self.env.observation_space.shape))#(T,[d_s])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)#(1,T)
#        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)#(1,T)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)#(1,T+1), the first done will be throw away because it is a default dummy value to use in model.step, not returned from env.step. Could be useful for lstm but lstm is not implemented here.
        
        a_label = np.asarray(a_label_episode,dtype = np.int32)#(T,1)
        prob_n = np.asarray(prob_episode,dtype = np.float32).reshape([-1])#flatten it,(sum of n_ac_candidates)
        all_a = np.asarray(all_a_episode,dtype = np.float32).reshape([-1] + list(d_action))#(sum of n_ac_candidates,d_a)
        num_a = np.asarray(num_a_list,dtype = np.int32)#(T)
        
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        #TODO if using buffer, this self.model.value will be late and should be changed
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()#(1,),since n_env is 1
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])#(T,)
#        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])#(sum of n_ac_candidates,)
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])#(masks is left 1 shift of dones, probably used for lstm, but haven't implemented here)
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])#(T,)
        
        #now use replay_buffer, only for n_env = 1
#        self.replay_buffer.store_frames(list(mb_obs),all_a_episode,prob_episode,?,?,a_label_episode,num_a_list,list(mb_dones),None,None)
        
        
        
        
        return mb_obs, mb_states, mb_rewards, mb_masks, None, mb_values, ep_infos, true_rewards,a_label,prob_n,all_a,num_a
