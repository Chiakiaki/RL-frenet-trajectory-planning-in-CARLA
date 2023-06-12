import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from matplotlib.font_manager import FontProperties
from scipy import stats
'''
Note that, the reward here is episode reward.


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
def make_collision_result_list(data,collision_r = -10):
    result_list = []
    for i in np.arange(len(data)):
        y = 0
        x_sum = 0
        result = []
        for j in np.arange(len(data[i]['l'])):
            x_sum = x_sum + data[i]['l'][j]
            if data[i]['r'][j] == collision_r:
                y = y+1
                result.append(x_sum)
        result = np.array(result)
        result_list.append(result)
    return result_list

def plot_collision2(result_list,c = '',label = None):
    [plt.plot(i,np.arange(len(i)),c) for i in result_list[:-1]]
    [plt.plot(i,np.arange(len(i)),c,label = label) for i in result_list[-1:]]
    
def plot_collision(result_list,c = '',label = None):
    [plt.plot(i,np.arange(len(i)),c,label = label) for i in result_list]

def plot_rewards(args,folder, window_size=100, colors=None, alpha=0.2, lr=None, n_timesteps=float('inf'), filter_list=[''], filter_full_dir=0):
    data = []
    
    filter_ = ''
    agents=[]
    color_use = []
    names = []
    average_for_t = []
    
    for ii,filter_i in enumerate(filter_list):
        folder[0]
        if filter_full_dir == 0:
            path = './logs/' + folder[0] + '/' + '*' + filter_i + '*/'
        else:
            path = './logs/' + folder[0] + '/' + filter_i + '/'
        agents_i = glob.glob(path)
        agents.append(agents_i[0])
        if colors is not None:#note: the "for" in python may run parallelly
            color_use.append(colors[ii])
        
        i = agents_i[0]
        if len(args.file_name) == 1:
            names.append(i.split('/')[-2])#the last folder
        for fn in args.file_name:
            if len(args.file_name) > 1:
                names.append(i.split('/')[-2]+'_'+fn)#the last folder
            data.append(pd.read_csv(i+'/'+fn, skiprows=1))

    agents = list(set(agents))
        

    

    



    average = []
    std_dev = []
    step_cum = []

    if lr is None:
        lr = names
        print(lr)
        for i,_ in enumerate(lr):
            lr[i]=lr[i].replace("_7en4","")
            lr[i]=lr[i].replace("shorthard_","")
            lr[i]=lr[i].replace("7en","7e-")

    if args.plot_collision == 0:
        for ii,x in enumerate(range(len(data))):
            temp = []
            temp_step = []
            temp_std = []
            sum_ = 0
    
            for i in range(window_size - 1):
                # temp.append(np.mean(data[x]['r'][:i]))
                temp.append(0)
                sum_ += data[x]['l'][i]
                temp_step.append(sum_)
            
            for i in range(window_size - 1, data[x]['r'].shape[0]):
                temp.append(np.mean(data[x]['r'][i - window_size - 1:i]))
#                temp_std.append(np.std(data[x]['r'][i - window_size - 1:i])/10)
                temp_std.append(np.std(data[x]['r'][i - window_size - 1:i]))
                sum_ += data[x]['l'][i]
                temp_step.append(sum_)
                if sum_ > n_timesteps:
                    break
    
            average.append(temp)
            std_dev.append(temp_std)
            step_cum.append(temp_step)

    
        plt.figure(figsize=(12, 8))        
        
            
        if colors is None:
            color_use = [np.random.rand(3, ) for x in names]
    

    
        for i in range(len(lr)):
            plt.plot(step_cum[i], average[i], '-', color=color_use[i])
            plt.fill_between(step_cum[i][window_size - 1:], np.subtract(average[i][window_size - 1:], std_dev[i]),
                             np.add(average[i][window_size - 1:], std_dev[i]), color=color_use[i], alpha=alpha)
    
        plt.title('CARLA')
        if folder == ["v5_10vehicles"]:
            pass
        else:
            plt.ylim([-14.5,1])
            plt.xlim([-1000,3300000])
        plt.xlabel('TimeSteps')
        plt.ylabel('Mean_Episode_Reward_-{}'.format(window_size))
        plt.legend(lr)
        plt.grid()
        plt.show()
    
        def find_first_ind_after_n(n,ind):
            for index,i in enumerate(ind):
                if i>=n:
                    return index
            print("error ind too large")
            return len(ind)-1
            
        #t_test
        if len(data) == 2:
            #lets do it at the 3000000
    
            
            ind1 = np.nanargmax(np.where(np.array(average[0]) != 0, average[0], np.nan))
            ind2 = np.nanargmax(np.where(np.array(average[1]) != 0, average[1], np.nan))
            
            mean_diff = average[0][ind1]-average[1][ind2]
            std1 = std_dev[0][ind1-window_size+1]
            std2 = std_dev[1][ind2-window_size+1]
            t1,p1 = t_test_for_mean(mean_diff,std1,std2,window_size)
            print('t:',t1,' p:',1-p1)
            print('best:',average[0][ind1],average[1][ind2])
            
            
            ind1 = find_first_ind_after_n(3000000,step_cum[0])
            ind2 = find_first_ind_after_n(3000000,step_cum[1])
            mean_diff = average[0][ind1]-average[1][ind2]
            std1 = std_dev[0][ind1-window_size+1]
            std2 = std_dev[1][ind2-window_size+1]
            t2,p2 = t_test_for_mean(mean_diff,std1,std2,window_size)
            print('t:',t2,' p:',1-p2)
            print('3000000:',average[0][ind1],average[1][ind2])        
    
        if 1:
        # the maximum average and at
            if args.query_ind == -1:
                if folder == ["v5_10vehicles"]:
                    query = 100000
                else:
                    query = 3000000
            else:
                query = args.query_ind

            for i in np.arange(len(data)):
                ind1 = find_first_ind_after_n(query,step_cum[i])
                print(lr[i],ind1,': ',average[i][ind1],'BestBefore: ',np.nanmax(average[i][:ind1]))


            
    else:
        result_list = make_collision_result_list(data)
        plt.figure(figsize=(12, 8))
        plot_collision(result_list,c = '',label = lr)
        plt.xlabel('TimeSteps')
        plt.ylabel('Total number of collision-{}'.format(window_size))
        plt.legend(lr)
        plt.grid()
        plt.show()            
        


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
    parser.add_argument('--filter_full_dir', type=int,default=0)
    parser.add_argument('--plot_collision', type=int,default=0)
    parser.add_argument('--file_name', nargs='+', type=str,default=["monitor.csv"])
    parser.add_argument('--query_ind', type=int, default=-1)
    args = parser.parse_args()
    plot_rewards(args, args.folder, args.window_size, args.colors, args.alpha, args.lr, args.n_steps, args.filter, args.filter_full_dir)
