import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from matplotlib.font_manager import FontProperties
from scipy import stats
from scipy.sparse import coo_matrix
'''
Please use this one. I fixed many bugs and
the computation is now correct.


Utility Folder in order to visualize Rewards
Usage: place agent folder in logs/agent_xx
args: 
--agent_ids: agents to be plotted (default = None)
--window_size: window size of moving (average default = 100)
--legends: : ['DDPG','TRPO']
--colors: list: ['red','blue']
--alpha: opacity
example usage:
python reward_plotter.py --agent_ids 26 27 28 --window_size=10
'''
font = FontProperties(fname=r"media\\sry\\OS\\Windows\\Fonts\\timesbd.ttf",size = 14)#C:\WINDOWS\Fonts
font2 = FontProperties(fname=r"media\\sry\\OS\\Windows\\Fonts\\timesbd.ttf",size = 20)#C:\WINDOWS\Fonts
plt.rcParams['font.family'] = font2.get_family()

plt.rcParams['figure.subplot.bottom']=0.2
plt.rcParams['figure.subplot.left']=0.2
plt.rcParams['figure.figsize'] = (4.0,3.5)
plt.rcParams['font.size'] = 15
plt.rcParams['lines.linewidth'] = 2.0

def t_test_for_mean(mean_diff,std1,std2,n,need_process_std=True):
    # one-tailed test. So, the null hyppothesis is that
    # the smaller mean samples has mean no less than the larger mean samples
    # scipy.state.ttest is two-tailed test, so will return doulbe
    mu = mean_diff
    std1=np.float32(std1)
    std2=np.float32(std2)
    if need_process_std:
        std1 = std1*np.sqrt(n/(n-1))
        std2 = std2*np.sqrt(n/(n-1))
    t = mu / np.sqrt(std1**2/n+std2**2/n)
    de = 1/(n-1)*(std1**2/n)**2 + 1/(n-1)*(std2**2/n)**2
    tdof = (std1**2/n+std2**2/n)**2 / de
    p = stats.t.sf(abs(t), df=tdof)
    return t,p
    
    
"""
debug for above t_test_for_mean

a = [1,1,3,3]
b = [1,1,5,5]
n=4
mean_diff = np.mean(a) - np.mean(b)
t,p = t_test_for_mean(mean_diff,std1,std2,4)
print(stats.ttest_ind(a,b))
"""
    
    
    #on use:
    #stats.ttest_ind(?,?)


def plot_rewards(folder, window_size=100, colors=None, alpha=0.2, lr=None, n_timesteps=float('inf'), filter_list=['']):
    data = []
    
    filter_ = ''
    agents=[]
    
    average_for_t = []
    
    for filter_i in filter_list:
        folder[0]
        path = './logs/' + folder[0] + '/' + '*' + filter_i + '*/'
        names = []
        agents_i = glob.glob(path)
        agents += agents_i
    agents = list(set(agents))
        
    for i in agents:
        names.append(i.split('/')[-2])#the last folder
    

    
    for i in agents:
        data.append(pd.read_csv(i+'/monitor.csv', skiprows=1))    

    average = []
    std_dev = []
    step_cum = []

    def find_first_ind_after_n(n,ind):
        for index_of_ind,i in enumerate(ind):
            if i>=n:
                return index_of_ind

    
    for x in range(len(data)):
        temp = []
        temp_step = []
        temp_std = []
        sum_ = 0
        
        
        
        for i in range(data[x]['r'].shape[0]):
            sum_ += data[x]['l'][i]
            temp_step.append(sum_)
            if sum_ > n_timesteps:
                break
            

        r_mat = coo_matrix((data[x]['r'], (np.zeros_like(temp_step), temp_step)), shape=(1, temp_step+1)).toarray()

        
        ind_of_ind = find_first_ind_after_n(window_size)
        
        for i in range(window_size - 1):
            # temp.append(np.mean(data[x]['r'][:i]))
            temp.append(0)

        for i in range(window_size - 1, data[x]['r'].shape[0]):
            temp.append(np.mean(data[x]['r'][i - window_size - 1:i]))
            temp_std.append(np.std(data[x]['r'][i - window_size - 1:i]))
            if sum_ > n_timesteps:
                break

        average.append(temp)
        std_dev.append(temp_std)
        step_cum.append(temp_step)

    plt.figure(figsize=(12, 8))

    if lr is None:
        lr = names
        print(lr)
        for i,_ in enumerate(lr):
            lr[i]=lr[i].replace("_7en4","")
            lr[i]=lr[i].replace("shorthard_","")
            lr[i]=lr[i].replace("7en","7e-")
        
        
    if colors is None:
        colors = [np.random.rand(3, ) for x in agents]

    for i in range(len(lr)):
        plt.plot(step_cum[i], average[i], '-', color=colors[i])
        plt.fill_between(step_cum[i][window_size - 1:], np.subtract(average[i][window_size - 1:], std_dev[i]),
                         np.add(average[i][window_size - 1:], std_dev[i]), color=colors[i], alpha=alpha)

    plt.title('CARLA')
#    plt.ylim([-25,75])
    plt.xlabel('TimeSteps')
    plt.ylabel('Mean_Reward-{}'.format(window_size))
    plt.legend(lr)
    plt.grid()
    plt.show()
    
    def find_first_ind_after_n(n,ind):
        for index,i in enumerate(ind):
            if i>=n:
                return index
        
    #t_test
#    if len(data) == 2:
#        #lets do it at the 3000000
#        ind1 = find_first_ind_after_n(3000000,step_cum[0])
#        ind2 = find_first_ind_after_n(3000000,step_cum[1])
#        mean_diff = average[0][ind1]-average[1][ind2]
#        std1 = std_dev[0][ind1]
#        std2 = std_dev[1][ind2]
#        t,p = t_test_for_mean(mean_diff,std1,std2,window_size)
#        print('t:',t,' p:',p)
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', nargs='+', type=str, default=["v5_10vehicles"])
#    parser.add_argument('--agent_ids', nargs='+', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--colors', nargs='+', type=str, default=None)
    parser.add_argument('--lr', nargs='+', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--n_steps', type=float, default=1e7)
    parser.add_argument('--filter', nargs='+', type=str, default=[''])
    args = parser.parse_args()
    plot_rewards(args.folder, args.window_size, args.colors, args.alpha, args.lr, args.n_steps, args.filter)
