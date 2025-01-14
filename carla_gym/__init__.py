from gym.envs.registration import register

register(
    id='CarlaGymEnv-v1',
    entry_point='carla_gym.envs:CarlaGymEnv_v1')

register(
    id='CarlaGymEnv-v2',
    entry_point='carla_gym.envs:CarlaGymEnv_v2')
    
register(
    id='CarlaGymEnv-v3',
    entry_point='carla_gym.envs:CarlaGymEnv_v3')
    
register(
    id='CarlaGymEnv-v4',
    entry_point='carla_gym.envs:CarlaGymEnv_v1_for_BDPL')
    
register(
    id='CarlaGymEnv-v5',
    entry_point='carla_gym.envs:CarlaGymEnv_v1_generic')
    
register(
    id='CarlaGymEnv-v6',
    entry_point='carla_gym.envs:CarlaGymEnv_v2_continuous')
