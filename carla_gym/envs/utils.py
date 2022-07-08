import numpy as np


def x2yaw(x,y,dt):
    dy = np.diff(y,axis = -1)
    dx = np.diff(x,axis = -1)
    v = np.sqrt(dx**2+dy**2)/dt
    yaw = np.arctan2(dy,dx)
    return v,yaw

def traj2action_old(traj):
    x = np.array([x.x for x in traj])#length T+2
    y = np.array([x.y for x in traj])
    v,yaw = x2yaw(x,y,1) #length T+1
    v = v[1:]
    yaw_change = yaw[1:] - yaw[0]#The difference is here
    return v,yaw_change

def traj2action(traj,T_ac_candidates,dt = 0.1,scale_yaw = 40,scale_v = 0.01):
    x = np.array(traj.x)#length T+2
    y = np.array(traj.y)
    v,yaw = x2yaw(x,y,dt) #length T+1
    v = v[:T_ac_candidates]
    yaw_change = yaw[1:T_ac_candidates+1] - yaw[:T_ac_candidates]#length 0
    return np.concatenate([yaw_change*scale_yaw,v*scale_v])

