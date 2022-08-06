import numpy as np

class Frenet_path_fake:
    #copy past
    def __init__(self):
#        self.id = None
#        self.t = []
#        self.d = []
#        self.d_d = []
#        self.d_dd = []
#        self.d_ddd = []
#        self.s = []
#        self.s_d = []
#        self.s_dd = []
#        self.s_ddd = []
#        self.cd = 0.0
#        self.cv = 0.0
#        self.cf = 0.0

        self.x = []
        self.y = []
        self.t = []
#        self.z = []
#        self.yaw = []
#        self.ds = []
#        self.c = []
#
#        self.v = []  # speed



def x2yaw(x,y,dt):
    dy = np.diff(y,axis = -1)
    dx = np.diff(x,axis = -1)
    v = np.sqrt(dx**2+dy**2)/dt
    yaw = np.arctan2(dy,dx)
    return v,yaw

def yaw2x(v,yaw,dt,x0,y0):
    assert len(np.shape(v)) == 1 and len(np.shape(yaw)) == 1,"yaw2x dim error"
    dx = v*np.cos(yaw)*dt
    dy = v*np.sin(yaw)*dt
    x = np.cumsum(np.concatenate([[x0],dx]))
    y = np.cumsum(np.concatenate([[y0],dy]))
    return x,y
    

def traj2action_old(traj):
    x = np.array([x.x for x in traj])#length T+2
    y = np.array([x.y for x in traj])
    v,yaw = x2yaw(x,y,1) #length T+1
    v = v[1:]
    yaw_change = yaw[1:] - yaw[0]#The difference is here
    return v,yaw_change

def rescure_yaw_change(yaw_change):
    """
    #e.g. 3.13 - 0 should be -0.1, not -3.13
    #output range from -pi to pi
    """
    return np.mod(yaw_change + np.pi,np.pi*2)-np.pi

class traj_action_params(object):
    def __init__(self,yaw0, T_ac_candidates, dt, scale_yaw = 40,scale_v = 0.01):
        self.dt = dt
        self.scale_yaw = scale_yaw
        self.scale_v = scale_v
        self.T_ac_candidates = T_ac_candidates
        self.yaw0 = yaw0
    
    def get_params(self):
        return self.yaw0,self.T_ac_candidates,self.scale_yaw,self.scale_v,self.dt
    
def traj2action(traj,traj_action_params):
    
    yaw0,T_ac_candidates,scale_yaw,scale_v,dt = traj_action_params.get_params()
    assert len(traj.x) > T_ac_candidates
    x = np.array(traj.x)#length at least T+1
    y = np.array(traj.y)
    v,yaw = x2yaw(x,y,dt)
    v = v[:T_ac_candidates]
    yaw = np.concatenate([[yaw0],yaw])
    yaw_change = yaw[1:T_ac_candidates+1] - yaw[:T_ac_candidates]
    yaw_change = rescure_yaw_change(yaw_change)
    return np.concatenate([yaw_change*scale_yaw,v*scale_v]),x[0],y[0]

def traj2action_no_start_yaw(traj,traj_action_params):
    
    yaw0,T_ac_candidates,scale_yaw,scale_v,dt = traj_action_params.get_params()
    assert len(traj.x) > T_ac_candidates + 1
    x = np.array(traj.x)#length at least T+1
    y = np.array(traj.y)
    v,yaw = x2yaw(x,y,dt)
    v = v[:T_ac_candidates]
    yaw = np.concatenate([[yaw0],yaw])
    yaw_change = yaw[2:T_ac_candidates+2] - yaw[1:T_ac_candidates+1]
    yaw_change = rescure_yaw_change(yaw_change)
    return np.concatenate([yaw_change*scale_yaw,v*scale_v]),x[0],y[0]

def get_traj_x0(traj):
    x = traj.x
    y = traj.y
    return x[0],y[0]

def action2traj(action,x0,y0,traj_action_params):
    
    yaw0,T_ac_candidates,scale_yaw,scale_v,dt = traj_action_params.get_params()
    assert len(np.shape(action)) == 1 and len(action) == T_ac_candidates*2

    yaw_change = action[:T_ac_candidates]/scale_yaw
    yaw = np.cumsum(np.concatenate([[yaw0],yaw_change]))[1:T_ac_candidates+1]
    v = action[-T_ac_candidates:]/scale_v
    x,y = yaw2x(v,yaw,dt,x0,y0)
    
    traj = Frenet_path_fake()
    traj.x = list(x)
    traj.y = list(y)
    traj.t = np.linspace(0,(T_ac_candidates-1)*dt,T_ac_candidates)

    return traj

def traj_distance_l2(traj1,traj2):
    #compute the l2 distance at the 'same-time end point'
    len1 = len(traj1.x)
    len2 = len(traj2.x)
    end = min(len1,len2) - 1
    dis = (traj1.x[end] - traj2.x[end])**2 + (traj1.y[end] - traj2.y[end])**2
    return dis
    


if __name__ == '__main__':
    #for test above
    traj = Frenet_path_fake()
    traj.x = np.random.rand(15)
    traj.y = np.random.rand(15)
    params = traj_action_params(0.5,14,0.1)
    ac,x0,y0 = traj2action(traj,params)
    traj2 = action2traj(ac,x0,y0,params)
    x1 = traj.x
    y1 = traj.y
    x2 = traj2.x
    y2 = traj2.y
    traj_t0 = get_traj_x0(traj)
    
    traj3 = Frenet_path_fake()
    traj3.x = np.random.rand(12)
    traj3.y = np.random.rand(12)
    
    dis_reconstruct = traj_distance_l2(traj,traj2)
    dis_test = traj_distance_l2(traj,traj3)
