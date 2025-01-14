"""
@author: Majid Moghadam
UCSC - ASL
"""

import gym
import numpy as np
import time
import math
import itertools
import matplotlib.pyplot as plt
from tools.modules import *
from config import cfg
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.local_planner.frenet_optimal_trajectory_mobil import FrenetPlanner as MotionPlanner_mobil
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel

from .utils import traj2action,action2traj,get_traj_x0,traj_action_params,traj_distance_l2,traj2action_no_start_yaw2

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN']


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index

def get_speed_ms(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)



class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self,**kwargs):
        """
        now we use kwargs to control action space
        mode: one  of bdp, ddpg, catagorical
        """
        self.mode = kwargs['mode']
        self.is_finish_traj = kwargs.pop('is_finish_traj',1)
        self.use_lidar = kwargs.pop('use_lidar',0)
        self.num_traj = kwargs.pop('num_traj', 3)
        self.scale_yaw = kwargs.pop('scale_yaw', 40)
        self.scale_v = kwargs.pop('scale_v',0.01)
        self.debug_bdp = kwargs.pop('debug',0)
        self.short_hard_mode = kwargs.pop('short_hard',0)
        self.env_change = kwargs.pop('env_change','None')#now can be 'pertubation'
        self.restart_every = 100
        self.step_counter = 0
        self.global_steps = 0
        self.is_save_log = False
        self.log_dir = None
        self.update_tf_list = 1
        assert self.num_traj % 3 == 0, "currently need num_traj as integer time of 3"
        

        
        
        self.__version__ = "9.9.2"

        # simulation
        self.verbosity = 0
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.time_step = int(cfg.GYM_ENV.TIME_STEP)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True

        # RL
        self.w_speed = int(cfg.RL.W_SPEED)
        self.w_r_speed = int(cfg.RL.W_R_SPEED)

        self.min_speed_gain = float(cfg.RL.MIN_SPEED_GAIN)
        self.min_speed_loss = float(cfg.RL.MIN_SPEED_LOSS)
        self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)

        self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)
        self.collision_penalty = int(cfg.RL.COLLISION)

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.low_state = np.array([[-1 for _ in range(self.look_back)] for _ in range(16)])
            self.high_state = np.array([[1 for _ in range(self.look_back)] for _ in range(16)])
        else:
            self.low_state = np.array(
                [[-1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
            self.high_state = np.array(
                [[1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
        
        if self.use_lidar == 0:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.time_step + 1, 9),
                                                dtype=np.float32)
        else:
            print('use lidar sensor for state representation')
            self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(362,),
                                                dtype=np.float32)
        
        
        self.T_ac_candidates = int(cfg.T_ACTION_CANDIDATES / cfg.CARLA.DT)#cfg.T_ACTION_CANDIDATES: length of trajectories used for action(secondes), must be less than 4.9
        
        if self.mode == 'bdp' or self.mode == 'ddpg':
            """Changes for BDPL: action space is ac_candidates's space"""
            """ddpg use same action space with bdp"""
            action_low = np.array([-1.]*(self.T_ac_candidates+1))#-1 because we use 'yaw_change' for action feature 
            action_high = np.array([1.]*(self.T_ac_candidates+1))
            self.action_space = gym.spaces.Box(low = action_low, high=action_high, shape=([self.T_ac_candidates+1]), dtype=np.float32)
        elif self.mode == 'bdpCatagorical':
            """still box as space, but only give one-hot label"""
            action_low = np.array([-1.]*(self.num_traj))#-1 because we use 'yaw_change' for action feature 
            action_high = np.array([1.]*(self.num_traj))
            self.action_space = gym.spaces.Box(low = action_low, high=action_high, shape=([self.num_traj]), dtype=np.float32)
        elif self.mode == 'combined':
            action_low = np.array([-1.]*(self.T_ac_candidates+1 + self.num_traj))#-1 because we use 'yaw_change' for action feature 
            action_high = np.array([1.]*(self.T_ac_candidates+1 + self.num_traj))
            self.action_space = gym.spaces.Box(low = action_low, high=action_high, shape=([self.T_ac_candidates+1 + self.num_traj]), dtype=np.float32)
        elif self.mode == 'continuous_catagorical':
            """original action space of env_v1"""
            action_low = np.array([-1])
            action_high = np.array([1])
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        elif self.mode == 'catagorical':
            """catagorical"""
            self.action_space = gym.spaces.Discrete(self.num_traj)
        elif self.mode == 'ddpg_on_params':
            """ddpg while action space is frenet params df and vf"""
            action_low = np.array([-1.,-1.])
            action_high = np.array([1.,1.])
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        elif self.mode == 'end2end':
            """directly compute the control command"""
            assert self.short_hard_mode == 1, "end to end mode must work in short hard mode"
            action_low = np.array([-1.])
            action_high = np.array([1.])
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        elif self.mode == 'mobil':
            """this mode use 'mobil' a rule-based behavior planning machanism"""
            pass

        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.zeros_like(self.observation_space.sample())

        # instances
        self.ego = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.acceleration_ = 0
        self.eps_rew = 0
        self.steer_pre = 0.0

        """
        ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """
        self.actor_enumerated_dict = {}
        self.actor_enumeration = []
        self.side_window = 5  # times 2 to make adjacent window

        self.motionPlanner = None
        self.vehicleController = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05
            
        #BDP use
        self.bdpl_path_list = None

    def seed(self, seed=None):
        pass
    
    def open_log_saver(self, log_dir, is_train = False):
        self.is_save_log = True
        self.log_dir = log_dir
        if is_train == True:
            self._log_appendix = '_train'
        else:
            self._log_appendix = '_test'
            

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """
        This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
        it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
        too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
        """
        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d

        init_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        target_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None # no vehicle ahead
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]
            init_s_idx = np.concatenate((np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
                                        , axis=1)
            sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]

            init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
            init_strict_s_idx = np.concatenate(
                (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]

            target_lane_s = np.array(others_s)[target_lane_d_idx]
            target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
                                                (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
            sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]

            if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            return self.traffic_module.actors_batch[vehicle_ahead_idx]['Actor']

    def enumerate_actors(self):
        """
        Given the traffic actors and ego_state this fucntion enumerate actors, calculates their relative positions with
        to ego and assign them to actor_enumerated_dict.
        Keys to be updated: ['LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """

        self.actor_enumeration = []
        ego_s = self.actor_enumerated_dict['EGO']['S'][-1]
        ego_d = self.actor_enumerated_dict['EGO']['D'][-1]

        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        others_id = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State']
            others_s[i] = act_s[-1]
            others_d[i] = act_d
            others_id[i] = actor['Actor'].id

        def append_actor(x_lane_d_idx, actor_names=None):
            # actor names example: ['left', 'leftUp', 'leftDown']
            x_lane_s = np.array(others_s)[x_lane_d_idx]
            x_lane_id = np.array(others_id)[x_lane_d_idx]
            s_idx = np.concatenate((np.array(x_lane_d_idx).reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1),
                                    x_lane_id.reshape(-1, 1)), axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (
                    any(abs(
                        sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window)) else -1)

        # --------------------------------------------- ego lane -------------------------------------------------
        same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) < 1)[0]
        if len(same_lane_d_idx) == 0:
            self.actor_enumeration.append(-2)
            self.actor_enumeration.append(-2)

        else:
            same_lane_s = np.array(others_s)[same_lane_d_idx]
            same_lane_id = np.array(others_id)[same_lane_d_idx]
            same_s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1),
                                         same_lane_id.reshape(-1, 1)), axis=1)
            sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()]
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])]
                                          if (any(sorted_same_s_idx[:, 1] > 0)) else -1)
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])]
                                          if (any(sorted_same_s_idx[:, 1] < 0)) else -1)

        # --------------------------------------------- left lane -------------------------------------------------
        left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]
        if ego_d < -1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(left_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(left_lane_d_idx)

        # ------------------------------------------- two left lane -----------------------------------------------
        lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -6.5) * ((np.array(others_d) - ego_d) > -7.5))[0]

        if ego_d < 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(lleft_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(lleft_lane_d_idx)

            # ---------------------------------------------- rigth lane --------------------------------------------------
        right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 3) * ((np.array(others_d) - ego_d) < 4))[0]
        if ego_d > 5.25:
            self.actor_enumeration += [-2, -2, -2]

        elif len(right_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(right_lane_d_idx)

        # ------------------------------------------- two rigth lane --------------------------------------------------
        rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 6.5) * ((np.array(others_d) - ego_d) < 7.5))[0]
        if ego_d > 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(rright_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(rright_lane_d_idx)

        # Fill enumerated actor values

        actor_id_s_d = {}
        norm_s = []
        # norm_d = []
        for actor in self.traffic_module.actors_batch:
            actor_id_s_d[actor['Actor'].id] = actor['Frenet State']

        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                actor_norm_s = []
                act_s_hist, act_d = actor_id_s_d[actor_id]  # act_s_hist:list act_d:float
                for act_s, ego_s in zip(list(act_s_hist)[-self.look_back:], self.actor_enumerated_dict['EGO']['S'][-self.look_back:]) :
                    actor_norm_s.append((act_s - ego_s) / self.max_s)
                norm_s.append(actor_norm_s)
            #    norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
            # -1:empty lane, -2:no lane
            else:
                norm_s.append(actor_id)

        # How to fill actor_s when there is no lane or lane is empty. relative_norm_s to ego vehicle
        emp_ln_max = 0.03 # Left_UP
        emp_ln_min = -0.03 # Left, Left_DOWN
        no_ln_down = -0.03  #
        no_ln_up = 0.004  #
        no_ln = 0.001  #

        if norm_s[0] not in (-1, -2):
            self.actor_enumerated_dict['LEADING'] = {'S': norm_s[0]}
        else:
            self.actor_enumerated_dict['LEADING'] = {'S': [emp_ln_max]}

        if norm_s[1] not in (-1, -2):
            self.actor_enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
        else:
            self.actor_enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}

        if norm_s[2] not in (-1, -2):
            self.actor_enumerated_dict['LEFT'] = {'S': norm_s[2]}
        else:
            self.actor_enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}

        if norm_s[3] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
        else:
            self.actor_enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}

        if norm_s[4] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
        else:
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}

        if norm_s[5] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT'] = {'S': norm_s[5]}
        else:
            self.actor_enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}

        if norm_s[6] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
        else:
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}

        if norm_s[7] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
        else:
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}

        if norm_s[8] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT'] = {'S': norm_s[8]}
        else:
            self.actor_enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}

        if norm_s[9] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
        else:
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}

        if norm_s[10] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
        else:
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}

        if norm_s[11] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
        else:
            self.actor_enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}

        if norm_s[12] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
        else:
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}

        if norm_s[13] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
        else:
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}

    def fix_representation(self):
        """
        Given the traffic actors fill the desired tensor with appropriate values and time_steps
        """
        self.enumerate_actors()

        self.actor_enumerated_dict['EGO']['SPEED'].extend(self.actor_enumerated_dict['EGO']['SPEED'][-1]
                                                     for _ in range(self.look_back - len(self.actor_enumerated_dict['EGO']['NORM_D'])))

        for act_values in self.actor_enumerated_dict.values():
            act_values['S'].extend(act_values['S'][-1] for _ in range(self.look_back - len(act_values['S'])))

        _range = np.arange(-self.look_back, -1, int(np.ceil(self.look_back / self.time_step)), dtype=int) # add last observation
        _range = np.append(_range, -1)

        lstm_obs = np.concatenate((np.array(self.actor_enumerated_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_enumerated_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_DOWN']['S'])[_range]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state
    
    def get_Tf_list(self,env_change):
        if self.short_hard_mode == 1:
            if self.env_change == 'pertubation_old':
                if self.update_tf_list == 1:#we wish the traj generation parameter is not changed too frequently
                    if self.num_traj == 3:
                        Vf_n_list = [-1]
                        Tf_list = list(np.sort(np.random.choice(np.arange(32,69), 5, replace=False)/10.))
                    elif self.num_traj == 9:
                        Vf_n_list = [-2,-1,0]
                        Tf_list = list(np.sort(np.random.choice(np.arange(32,69), 5, replace=False)/10.))
                    elif self.num_traj == 15:
                        Vf_n_list = [-3,-2,-1,0,1]
                        Tf_list = list(np.sort(np.random.choice(np.arange(32,69), 5, replace=False)/10.))
                    self.Tf_list = Tf_list
                    self.update_tf_list = 0
                else:
                    Tf_list = self.Tf_list
            elif self.env_change == 'pertubation':
                if self.update_tf_list == 1:#we wish the traj generation parameter is not changed too frequently
                    if self.num_traj == 3:
                        Vf_n_list = [-1]
                        Tf_list = list(np.random.choice(np.arange(32,69), 1, replace=False)/10.)#do not use sort, as it show no difference
                    elif self.num_traj == 9:
                        Vf_n_list = [-2,-1,0]
                        Tf_list = list(np.random.choice(np.arange(32,69), 3, replace=False)/10.)
                    elif self.num_traj == 15:
                        Vf_n_list = [-3,-2,-1,0,1]
                        Tf_list = list(np.random.choice(np.arange(32,69), 5, replace=False)/10.)
                    self.Tf_list = Tf_list
                    self.update_tf_list = 0
                else:
                    Tf_list = self.Tf_list
                
                
            elif self.env_change == 'None':
                if self.num_traj == 3:
                    Vf_n_list = [-1]
                    Tf_list = [3.2]
                elif self.num_traj == 9:
                    Vf_n_list = [-2,-1,0]
                    Tf_list = [3.2,4.1,5]
                elif self.num_traj == 15:
                    Vf_n_list = [-3,-2,-1,0,1]
                    Tf_list = [3.2,4.1,5,5.9,6.8]
            else:
                raise NotImplementedError

                            
        else:
            if self.num_traj == 3:
                Vf_n_list = [-1]
                Tf_list = [5]
            elif self.num_traj == 9:
                Vf_n_list = [-2,-1,0]
                Tf_list = [4.2,5,5.8]
            elif self.num_traj == 15:
                if self.debug_bdp == 0:
                    Vf_n_list = [-3,-2,-1,0,1]
                    Tf_list = [3.2,4.1,5,5.9,6.8]
                else:
                    Vf_n_list = [-1,-1,-1,-1,-1]
                    Tf_list = [5,5,5,5,5]
            else:
                raise NotImplementedError
        return Tf_list

    def external_sampler(self):
        """
        Split the previous 'step' function into two part. The first part generated trajectories without action,
        the second part select the respective trajectory to the input action.
        So, 'external_sampler()' and 'step()' need to be called in turn. (Firstly this one, then 'step())
        """
        self.n_step += 1
        #TODO: is first path not treated
        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        if self.verbosity: print('ACTION'.ljust(15), '{:+8.6f}'.format(float(action)))
        if self.is_first_path:  # Episode start is bypassed
            self.is_first_path = False
            action = 0

        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """

        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
#        fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action, Tf=5,
#                                                                         Vf_n=-1)#original code
        
        """#wait after test
        bdpl_path_list=[]
        _, bdpl_path_list1= self.motionPlanner.run_step_without_update_self_path(ego_state, self.f_idx,0)#target speed not set
        _, bdpl_path_list2 = self.motionPlanner.run_step_without_update_self_path(ego_state, self.f_idx,-1)#target speed not set
        _, bdpl_path_list3 = self.motionPlanner.run_step_without_update_self_path(ego_state, self.f_idx,1)#target speed not set
        bdpl_path_list = []+bdpl_path_list1+bdpl_path_list2+bdpl_path_list3
        #TODO: set off the road
        off_the_road = False
        """
        def cal_path_list_with_off_road(self,ego_state, Vf_n = -1, Tf = 5):
            bdpl_path_list = []
            path1, lanechange1, off_the_road1 = self.motionPlanner.run_step_single_path_without_update_self_path_with_off_road(ego_state, self.f_idx, df_n=0, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code    
            path2, lanechange2, off_the_road2 = self.motionPlanner.run_step_single_path_without_update_self_path_with_off_road(ego_state, self.f_idx, df_n=-1, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code
            path3, lanechange3, off_the_road3 = self.motionPlanner.run_step_single_path_without_update_self_path_with_off_road(ego_state, self.f_idx, df_n=1, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code
            bdpl_path_list = [path1,path2,path3]
            return bdpl_path_list
        
        def cal_path_list_without_off_road(self,ego_state, Vf_n = -1, Tf = 5):
            # Will correct off road lanechange to go straight. This is the original logic.
            bdpl_path_list = []
            path1, lanechange1, off_the_road1 = self.motionPlanner.run_step_single_path_without_update_self_path(ego_state, self.f_idx, df_n=0, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code    
            path2, lanechange2, off_the_road2 = self.motionPlanner.run_step_single_path_without_update_self_path(ego_state, self.f_idx, df_n=-1, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code
            path3, lanechange3, off_the_road3 = self.motionPlanner.run_step_single_path_without_update_self_path(ego_state, self.f_idx, df_n=1, Tf=Tf,
                                                                             Vf_n=Vf_n)#original code
            bdpl_path_list = [path1,path2,path3]
            tmp_lanechange = [lanechange1,lanechange2,lanechange3]
            tmp_off_the_road = [off_the_road1,off_the_road2,off_the_road3]
            return bdpl_path_list, tmp_lanechange, tmp_off_the_road
        
        Tf_list = self.get_Tf_list(self.env_change)
        
        self.bdpl_path_list = []
        self.tmp_lanechange = []
        self.tmp_off_the_road = []
        self.bdpl_path_list_with_offroad = []
        
        for Tf in Tf_list:
            tmp_bdpl_path_list, tmp_tmp_lanechange, tmp_tmp_off_the_road = cal_path_list_without_off_road(self,ego_state,Tf = Tf)
            tmp_bdpl_path_list_with_offroad = cal_path_list_with_off_road(self,ego_state,Tf = Tf)
            self.bdpl_path_list += tmp_bdpl_path_list
            self.tmp_lanechange += tmp_tmp_lanechange
            self.tmp_off_the_road += tmp_tmp_off_the_road
            self.bdpl_path_list_with_offroad += tmp_bdpl_path_list_with_offroad
        
        
        #save some parames
                #some variable defined in external_sampler
#        traj_action_params1 = traj_action_params(psi,self.T_ac_candidates,dt = self.dt)
        traj_action_params1 = traj_action_params(psi,self.T_ac_candidates,dt = self.dt, scale_yaw = self.scale_yaw, scale_v = self.scale_v)
        self.external_sampler_variable = temp,init_speed,traj_action_params1,ego_state

        # convert path in bdpl_path_list to vector
        
        #note, the path.x and path.y are the only information in global coordinates
        #the path.yaw is the yaw of s-axis of the frenet coordinate (the global path), so they are the same
        #path.s,path.s_d,path.s_dd are also the same because they are of frenet coordinate


        def process_path_list_as_acs(path_list,traj_action_params):
            #note: when on vehicle start from stationary, the frenet trajectory for left lane-change and right
            #       lane change has sharp turning rate at start. So we neglect the first yaw_change value
            ac_candidates= [traj2action_no_start_yaw2(tmp_path,traj_action_params)[0] for tmp_path in path_list]
            return np.array(ac_candidates)
        
        

        if self.mode == 'bdp':
            #tmp_debug
            #note: when on vehicle start from stationary, the frenet trajectory for left lane-change and right
            #       lane change has sharp turning rate at start.
#            ac1 = process_path_list_as_acs(self.bdpl_path_list,traj_action_params1)
            ac2 = process_path_list_as_acs(self.bdpl_path_list_with_offroad,traj_action_params1)
#            if np.max(np.abs(ac2[:,0])) > 1:
#                print('e')
            
            return ac2,len(self.bdpl_path_list)
        elif self.mode == 'combined':
            ac2 = process_path_list_as_acs(self.bdpl_path_list_with_offroad,traj_action_params1)
            ac2 = np.concatenate([ac2,np.eye(3)],axis = 1)
            return ac2,len(self.bdpl_path_list)
        elif self.mode == 'continuous_catagorical':
            return np.array([[0.],[-1.],[1.]]),3#must be correctly set, is related to lane-change reward
            #note bdp does not implemented continuous catagorical
        elif self.mode == 'bdpCatagorical':
            return np.eye(3),3


        
    def step(self, action=None):
        #for dev
#        print(action)
        if self.mode == 'mobil':
            # non learning
            self.n_step += 1
            
            temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
            speed = get_speed(self.ego)
            acc_vec = self.ego.get_acceleration()
            acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
            psi = math.radians(self.ego.get_transform().rotation.yaw)
            ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp,self.max_s]
            # fpath = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])
            Tf_list = self.get_Tf_list(self.env_change)
            fpath, fplist, best_path_idx, self.lane_change = self.motionPlanner.run_step(ego_state, self.f_idx, self.traffic_module.actors_batch, target_speed=self.targetSpeed,Tf_list=Tf_list)
            
            init_speed = speed = get_speed(self.ego)
            off_the_road = False
        else:
            # learning
            if self.bdpl_path_list is None:
                # so external_sampler not called, call it now
                # to be compatible with original.
                self.external_sampler()
                
            if self.is_finish_traj == 0:# could drive off the road in such mode
                self.bdpl_path_list = self.bdpl_path_list_with_offroad
                
            temp,init_speed,traj_action_params1,ego_state = self.external_sampler_variable
            """
            Though action space is box, here the action passed in should be int, because,
            in Boltzmann Distribution Policy Learning, the env.step need not return all infomation to candidate actions,
            they just need to return some feature. The model just do 'select', not create.
            """
            if self.mode == 'catagorical' or self.mode == 'bdp' or self.mode == 'bdpCatagorical' or self.mode == 'combined':
                assert type(action) is int or np.int32 or np.int64, "error action %d is not type int" % action
                fpath = self.bdpl_path_list[action]
                self.lanechange = self.tmp_lanechange[action]
                off_the_road = self.tmp_off_the_road[action]
                
                """
                #tmp_test_for_reconstruct_action
                tmp_ac = traj2action(fpath,traj_action_params1)
                x0,y0 = get_traj_x0(self.bdpl_path_list[0])#get the x0,y0
                fpath_reconstruct = action2traj(tmp_ac,x0,y0,traj_action_params1)
                dis = traj_distance_l2(fpath,fpath_reconstruct)
                assert dis < 0.001
                """
            elif self.mode == 'ddpg_on_params':
                #simply calculate one with action
                fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path_without_update_self_path_continous_df(ego_state, self.f_idx, df_n=action[0]*1.5, Tf=5, Vf_n=action[1] -1)
    
    
    
                #TODO: change  Vf_n, remove the '-1'
            elif self.mode == 'ddpg':
                #action is of the same shape with
                x0,y0 = get_traj_x0(self.bdpl_path_list[0])#get the x0,y0
                fpath = action2traj(action,traj_action_params1)
                raise NotImplementedError#fpath does not have frenet state
                
                #an action of time length 1 will yield a trajector with lenth 2.
                assert fpath.t == 5./self.dt
                
                #now judge whether it is lane_change
                dis = [traj_distance_l2(fpath,path2) for path2 in self.bdpl_path_list_with_offroad]
                nearest_idx = np.argmin(dis)
                self.lanechange = self.tmp_lanechange[nearest_idx]
                off_the_road = self.tmp_off_the_road[nearest_idx]
                
            elif self.mode == 'continuous_catagorical':
                assert type(action) is np.float32 or np.float64, "error action %d is not type float" % action
                #this code is previously inside motion planner, we move it here for equivalent
                if action < -0.33:
                    action = 1
                elif action > 0.33:
                    action = 2
                else:
                    action = 0
                fpath = self.bdpl_path_list[action]
                self.lanechange = self.tmp_lanechange[action]
                off_the_road = self.tmp_off_the_road[action]
            elif self.mode == 'end2end':
                assert len(self.bdpl_path_list) == 3,"for fair comparison, end to end is a lazy implementation. use 3 traj to compute the vehicle_ahead"
                action = action[0]
                if action < -1./3:
                    fpath = self.bdpl_path_list[1]#left
                    off_the_road = self.tmp_off_the_road[1]
                elif action > 1./3:
                    fpath = self.bdpl_path_list[2]#right
                    off_the_road = self.tmp_off_the_road[2]
                else:
                    fpath = self.bdpl_path_list[0]
                    off_the_road = self.tmp_off_the_road[0]
            else:
                raise NotImplementedError
            self.motionPlanner.update_self_path(fpath)
        
        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        
        self.f_idx = 1

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """

        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        # follows path until end of WPs for max 1.5 * path_time or loop counter breaks unless there is a langechange
        loop_counter = 0

        while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5 or
                                          loop_counter < self.loop_break or self.lanechange):# like during wps_to_go time of step, no replanning will do

            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwrite command speed using IDM
            ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
            ego_d = fpath.d[self.f_idx]
            vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            
            if self.mode == "end2end":
                control.steer = float(action)
            
            self.steer_pre = control.steer
            self.ego.apply_control(control)  # apply control

            """
                    **********************************************************************************************************************
                    *********************************************** Draw Waypoints *******************************************************
                    **********************************************************************************************************************
            """
            self.world_module.points_to_draw={}
            def add_draw_path(fpath,name_prefix='',color='COLOR_ALUMINIUM_0'):
                for i in range(len(fpath.t)):
                    self.world_module.points_to_draw[name_prefix+'path wp {}'.format(i)] = [
                        carla.Location(x=fpath.x[i], y=fpath.y[i]),
                        color]
            if self.world_module.args.play_mode != 0:
                if self.bdpl_path_list is not None:
                    for (j,i) in enumerate(self.bdpl_path_list_with_offroad):
                        add_draw_path(i,'path list {}'.format(j),color='COLOR_ALUMINIUM_0')
                add_draw_path(fpath,'',color='COLOR_ORANGE_0')
                    
                    
                    
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])
            
            
            
            """
            if self.world_module.args.play_mode != 0:
#                for i in range(len(fpath.t)):
                for i in range(self.T_ac_candidates):
                    self.world_module.points_to_draw['path wp {}'.format(i)] = [
                        carla.Location(x=fpath.x[i], y=fpath.y[i]),
                        'COLOR_ALUMINIUM_0']
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])
            """
            
            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            self.module_manager.tick()  # Update carla world
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()

            self.actor_enumerated_dict['EGO']['S'].append(ego_s)
            self.actor_enumerated_dict['EGO']['D'].append(ego_d)
            self.actor_enumerated_dict['EGO']['NORM_S'].append((ego_s - self.init_s) / self.track_length)
            self.actor_enumerated_dict['EGO']['NORM_D'].append(round((ego_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2))
            last_speed = get_speed(self.ego)
            self.actor_enumerated_dict['EGO']['SPEED'].append(last_speed / self.maxSpeed)
            # if ego off-the road or collided
            if any(collision_hist):
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True
                
            if self.is_finish_traj != 1:
                # in such mode, each env step only do the step of planned trajectory
                break
        
        
        self.bdpl_path_list = None#erase
        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        if self.use_lidar == 0:
            if cfg.GYM_ENV.FIXED_REPRESENTATION:
                self.state = self.fix_representation()# (5,9)
                if self.verbosity == 2:
                    print(3 * '---EPS UPDATE---')
                    print(TENSOR_ROW_NAMES[0].ljust(15),
                          #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
                         '{:+8.6f}'.format(self.state[-1][0]))
                    for idx in range(1, self.state.shape[1]):
                        print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))
    
    
            if self.verbosity == 3: 
                print(self.state)
            """*********************************************************************************************************************
                    *********************************************** RL Observation for lidar sensor *********************
                    *********************************************************************************************************************
            """
        else:
            d = self.world_module.lidar_sensor.d
            v = get_speed_ms(self.ego)
            rl_s_ds = np.concatenate([2 - d*2/(d+10),[v/20.],[self.steer_pre*10]])#old #(362,) assume this is the projection to front window
            self.state = rl_s_ds
#            plt.plot(rl_s_ds)
#            plt.show()
#            print(v,self.steer_pre)

        
        
        
        """
                **********************************************************************************************************************
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        if self.is_finish_traj == 1:#these 'lane_change' reward does not make sense in one_step mode
            e_speed = abs(self.targetSpeed - last_speed)
            r_speed = self.w_r_speed * np.exp(-e_speed ** 2 / self.maxSpeed * self.w_speed)  # 0<= r_speed <= self.w_r_speed
            #  first two path speed change increases regardless so we penalize it differently
    
            spd_change_percentage = (last_speed - init_speed) / init_speed if init_speed != 0 else -1
            r_laneChange = 0
        
            if self.lanechange and spd_change_percentage < self.min_speed_gain:
                r_laneChange = -1 * r_speed * self.lane_change_penalty  # <= 0
    
            elif self.lanechange:
                r_speed *= self.lane_change_reward
                
            positives = r_speed
            negatives = r_laneChange
            reward = positives + negatives  # r_speed * (1 - lane_change_penalty) <= reward <= r_speed * lane_change_reward
            # print(self.n_step, self.eps_rew)
                
        else:
            # only speed reward here.
            # TODO: However, the speed reward will receive at each step, and one episode has much more steps now, so the magnitude of speed reward need to be cut, at about $length_of_traj$ magnitude
            e_speed = abs(self.targetSpeed - last_speed)
            r_speed = self.w_r_speed * np.exp(-e_speed ** 2 / self.maxSpeed * self.w_speed)  # 0<= r_speed <= self.w_r_speed
            #  first two path speed change increases regardless so we penalize it differently
#            reward = (r_speed - self.w_r_speed * 0.5) / 100
#            if last_speed < self.targetSpeed * 1./2.:
#                reward = -1# now simply turn off speed reward, there is only collision reward now
#            else:
#                reward = 0
            reward = 0

        """
                **********************************************************************************************************************
                ********************************************* Episode Termination ****************************************************
                **********************************************************************************************************************
        """


        done = False
        if collision:
            # print('Collision happened!')
            reward = self.collision_penalty
            done = True
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
            
            if self.is_save_log == True:
                with open(self.log_dir + '/collision' + self._log_appendix + '.txt', "at") as fc:#open as appending mode
                    np.savetxt(fc, [self.global_steps],fmt='%d')
            
            
            return self.state, reward, done, {'reserved': 0}

        elif track_finished:
            # print('Finished the race')
            # reward = 10
            done = True
            if off_the_road:#TODO: to be deleted for shorthard mode
                reward = self.off_the_road_penalty
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
            return self.state, reward, done, {'reserved': 0}

        elif self.is_finish_traj:
            if off_the_road: #does not consider off road when doing one_step
                # print('off road happened!')
                reward = self.off_the_road_penalty
                # done = True
                self.eps_rew += reward
                # print('eps rew: ', self.n_step, self.eps_rew)
                if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
                return self.state, reward, done, {'reserved': 0}

        # reset the env for short hard mode
        self.step_counter += 1
        self.global_steps += 1
        if self.step_counter > self.restart_every and self.short_hard_mode == 1:

            done = True

        self.eps_rew += reward
        # print(self.n_step, self.eps_rew)
        if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
        return self.state, reward, done, {'reserved': 0}
    

    def reset(self):
        self.update_tf_list = 1
        
        self.step_counter = 0
        self.init_s = self.world_module.init_s
        
        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        
        if self.short_hard_mode == 1:
#            self.vehicleController.reset()
            self.world_module.reset(init_velocity = cfg.LOCAL_PLANNER.MIN_SPEED)
#            self.world_module.reset(init_velocity = 10)
            self.init_s = self.world_module.init_s
            init_d = self.world_module.init_d
            self.traffic_module.reset2(self.init_s, init_d)
        else:
            self.vehicleController.reset()
            self.world_module.reset()
            self.init_s = self.world_module.init_s
            init_d = self.world_module.init_d
            self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0
        self.steer_pre = 0.0

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True
        init_norm_d = round((init_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
        ego_s_list = [self.init_s for _ in range(self.look_back)]
        ego_d_list = [init_d for _ in range(self.look_back)]

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [0], 'NORM_D': [init_norm_d],
                                             'S': ego_s_list, 'D': ego_d_list, 'SPEED': [0]}
        if self.use_lidar == 0:
            if cfg.GYM_ENV.FIXED_REPRESENTATION:
                self.state = self.fix_representation()
                if self.verbosity == 2:
                    print(3 * '---RESET---')
                    print(TENSOR_ROW_NAMES[0].ljust(15),
                          #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
                          '{:+8.6f}'.format(self.state[-1][0]))
                    for idx in range(1, self.state.shape[1]):
                        print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))
        else:
            d = self.world_module.lidar_sensor.d
            v = get_speed_ms(self.ego)
            rl_s_ds = np.concatenate([2 - d*2/(d+10),[v/20.],[self.steer_pre]])#old #(362,) assume this is the projection to front window
            self.state = rl_s_ds
        
        for _ in range(1):#this seems to be the better solution for the first unstable frame.
            #I think this is probably because, the vehicle is spawn at e.g. 0.1 meters above the ground. So, it will fall at first.
            self.module_manager.tick()
            
        return self.state

    def begin_modules(self, args):
        self.verbosity = args.verbosity

        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)
        
        if self.mode == 'mobil':
            self.motionPlanner = MotionPlanner_mobil()
        else:
            self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        
        if self.short_hard_mode:
            self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0},args_longitudinal={'K_P': 1.0,
            'K_D': 0,
            'K_I': 1})
        else:
            self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
            
        
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()  # Update carla world

        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()
