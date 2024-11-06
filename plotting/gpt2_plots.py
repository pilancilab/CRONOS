import matplotlib.pyplot as plt
#import seaborn
import numpy as np

#For plotting
#plt.style.use("seaborn-v0_8")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Lucida Bright']

def plot_tuned(results, directory):
    grid_length, _ = results['AdamW']['test_acc'].shape
    t = 0
    Idx = results['AdamW']['best_lr_idx']
    traj = results['AdamW']['test_acc'][Idx]
    for i in range(grid_length):
       t_traj = np.cumsum(results['AdamW']['times'][i])
       t+=t_traj
    
    plt.plot(t, traj, label = 'Tuned AdamW')
    plt.plot(np.cumsum(results['CRONOS']['times']),results['CRONOS']['val_acc'], label = 'CRONOS', color = 'black', linestyle = '--')

    #plt.xlim(0, 1)     
    plt.xlabel("Time(s)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(directory, format='pdf')


def plot_trajs(results):
    grid_length, _ = results['AdamW']['test_acc'].shape
    lr_grid = np.sort(results['AdamW']['lr_grid'])
    for i in range(grid_length):
         traj = results['AdamW']['test_acc'][i]
         lr = lr_grid[i]
         plt.plot(traj, label = f'AdamW: lr = {lr}')
    
    plt.plot(results['CRONOS']['val_acc'], label = 'CRONOS', color = 'black', linestyle = '--')

    #plt.xlim(0, 1)     
    plt.xlabel("Passes/Outer Iterations")
    plt.ylabel("Validation Accuracy")
    plt.legend()
