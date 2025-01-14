"""
Created on Thu Oct 20 10:52:45 2022
start from copy past utils.py in trpo

remakrs:
"""


import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False, callback=None):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :param callback: (BaseCallback)
    :return: (dict) generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
        - continue_training: (bool) Whether to continue training
            or stop early (triggered by the callback)
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
#    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
#    actions = np.array([action for _ in range(horizon)])
    actions = np.zeros(horizon, 'int32')
    states = policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False
    
    #add 3 more return for BDPL
    a_label_episode, all_a_episode, num_a_list = [],[],[]

    callback.on_rollout_start()

    while True:
        # remark: input for bd policy is  (obs, ac_candidates, state=None, mask=None)
        # remakrs: currently only support one env
        ac_candidates,n_ac_candidates = env.env.external_sampler()#ac_candidates:(n,d)
        assert len(np.shape(ac_candidates)) == 2, "ac_candidates' dims should be 2"
        # remark: observation.reshape(-1, *observation.shape) will add a dim before observation.shape
        # remark: action here is actually action_idx,
        # original return in trpo using catacorical is of shape (1,), (1,), (None)
        action, vpred, states, prob = policy.step(observation.reshape(-1, *observation.shape),ac_candidates, states, done)
#        print(prob)
        action = action[0]#now shape (1,)
        d_action = np.shape(ac_candidates[[0],:])[1:]
        # TODO: check action shape
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        # remarks: add two more return
        if step > 0 and step % horizon == 0:
            callback.on_rollout_end()
            a_label = np.asarray(a_label_episode,dtype = np.int32)#(T,1)
            all_a = np.asarray(all_a_episode,dtype = np.float32).reshape([-1] + list(d_action))#(sum of n_ac_candidates,d_a)
            num_a = np.asarray(num_a_list,dtype = np.int32)#(T)
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,#(n,)
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                    'continue_training': True,
                    "all_a": all_a,
                    "num_a": num_a,
                    "a_label":a_label#(n,1)
            }
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape),ac_candidates)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            
            a_label_episode, all_a_episode, num_a_list = [],[],[]
            # Reset current iteration length
            current_it_len = 0
            callback.on_rollout_start()
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start
        
        all_a_episode.append(ac_candidates)#(T,)
        num_a_list.append(n_ac_candidates)#(T,n,d)
        a_label_episode.append(action)#(T,1)

        clipped_action = action
        # remarks: in bdp, the action is actually action_idx, does not need clipping
#        if isinstance(env.action_space, gym.spaces.Box):
#            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            #remarks: what is this?
            raise NotImplementedError
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward

        if callback is not None:
            if callback.on_step() is False:
                # We have to return everything so pytype does not complain
                a_label = np.asarray(a_label_episode,dtype = np.int32)#(T,1)
                all_a = np.asarray(all_a_episode,dtype = np.float32).reshape([-1] + list(d_action))#(sum of n_ac_candidates,d_a)
                num_a = np.asarray(num_a_list,dtype = np.int32)#(T)
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                    'continue_training': False,
                    "all_a": all_a,
                    "num_a": num_a,
                    "a_label":a_label
                    }
                return

        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
