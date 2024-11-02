import matplotlib.pyplot as plt
import seaborn
import numpy as np

#For plotting
plt.style.use("seaborn-v0_8")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Lucida Bright']


def plot_metric(solvers_metrics, metric, parameter, dir):
    if metric not in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
        raise ValueError("This metric is not supported!")
    for solver in solvers_metrics:
        data = solvers_metrics[solver]
        grid_length, n_epochs = data[metric].shape
        if solver !='Cronos_AM':
          for i in range(grid_length):
            if metric in ['training_loss', 'test_loss']:
                if parameter == 'passes':
                  plt.semilogy(np.arange(n_epochs)+1, data[metric][i], label = solver+':'+" "+'lr ='+" "+repr(data['lr_grid'][i]), 
                         linewidth=2.5)
                  plt.xlabel("Data passes")
                elif parameter == 'time':
                  plt.semilogy(np.cumsum(data['times'][i]), data[metric][i], label = solver+':'+" "+'lr ='+" "+repr(data['lr_grid'][i]), 
                  linewidth=2.5)
                  plt.xlabel("Time (s)")
            else:
              if parameter == 'passes':
                  plt.plot(np.arange(n_epochs)+1, data[metric][i], label = 'CRONOS-AM'+':'+" "+'lr ='+" "+repr(data['lr_grid'][i]), 
                         linewidth=2.5)
                  plt.xlabel("Data passes")
              elif parameter == 'time':
                  plt.plot(np.cumsum(data['times'][i]), data[metric][i], label = 'CRONOS-AM'+':'+" "+'lr ='+" "+repr(data['lr_grid'][i]), 
                  linewidth=2.5)
                  plt.xlabel("Time (s)")
        else:
          if metric in ['training_loss', 'test_loss']:
            
            if parameter == 'passes':
              x = 2*np.arange(n_epochs)
              plt.semilogy(x, data[metric][0], 
              label = solver+" "+'(ours)',
              color = 'goldenrod', linewidth=2.5)
              plt.xlabel("Data passes")
            
            elif parameter == 'time':
              plt.semilogy(np.cumsum(data['times'][0]), data[metric][0], 
              label = solver+" "+'(ours)',
              color = 'goldenrod', linewidth=2.5)
              plt.xlabel("Time (s)")
          else:
              if parameter == 'passes':
                x = 2*np.arange(n_epochs)
                plt.plot(x, data[metric][0], 
                label = solver+" "+'(ours)', 
                color = 'goldenrod', linewidth=2.5)
                plt.xlabel("Data passes")
              
              elif parameter == 'time':
                plt.plot(np.cumsum(data['times'][0]), data[metric][0], 
                label = solver+" "+'(ours)',
                color = 'goldenrod', linewidth=2.5)
                plt.xlabel("Time (s)")        
    if metric == 'train_loss':
        plt.ylabel("Training loss")
    
    elif metric == 'test_loss':
        plt.ylabel("Validation Loss")
    
    elif metric == 'train_acc':
        plt.ylabel("Training accuracy")
    
    else: 
        plt.ylabel("Validation Accuracy")

    plt.legend(title='Solver', frameon=True)


def plot_best_metric(solvers_metrics, metric,  
                     parameter, colors, dir):
    if metric not in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
        raise ValueError("This metric is not supported!")
    
    if parameter not in ['passes', 'time']:
        raise ValueError("This parameter is not supported!")
    
    for solver in solvers_metrics:
        
        data = solvers_metrics[solver]
        Idx = data['best_lr_idx']
        n_epochs = data[metric][Idx].shape[0]
        
        if solver != 'Cronos_AM':
          if metric in ['training_loss', 'test_loss']:
            
            if parameter == 'passes':
              plt.semilogy(np.arange(n_epochs)+1, data[metric][Idx], 
              label = solver+" "+'(tuned)',
              color = colors[solver], linewidth=2.5)
              plt.xlabel("Data passes")
            
            elif parameter == 'time':
              plt.semilogy(np.cumsum(data['times'][Idx]), data[metric][Idx], 
              label = solver+" "+'(tuned)',
              color = colors[solver], linewidth=2.5)
              plt.xlabel("Time (s)")

          else:
              
              if parameter == 'passes':
                plt.plot(np.arange(n_epochs)+1, data[metric][Idx], 
                label = solver+" "+'(tuned)', 
                color = colors[solver], linewidth=2.5)
                plt.xlabel("Data passes")
              
              elif parameter == 'time':
                plt.plot(np.cumsum(data['times'][Idx]), data[metric][Idx], 
                label = solver+" "+'(tuned)',
                color = colors[solver], linewidth=2.5)
                plt.xlabel("Time (s)")
        else:
          if metric in ['training_loss', 'test_loss']:
            
            if parameter == 'passes':
              x = 2*np.arange(n_epochs)
              plt.semilogy(x, data[metric][Idx], 
              label = 'CRONOS-AM'+" "+'(ours)',
              color = colors[solver], linewidth=2.5)
              plt.xlabel("Data passes")
            
            elif parameter == 'time':
              plt.semilogy(np.cumsum(data['times'][Idx]), data[metric][Idx], 
              label ='CRONOS-AM'+" "+'(ours)',
              color = colors[solver], linewidth=2.5)
              plt.xlabel("Time (s)")
          else:
              if parameter == 'passes':
                x = 2*np.arange(n_epochs)
                plt.plot(x, data[metric][Idx], 
                label = 'CRONOS-AM'+" "+'(ours)', 
                color = colors[solver], linewidth=2.5)
                plt.xlabel("Data passes")
              
              elif parameter == 'time':
                plt.plot(np.cumsum(data['times'][Idx]), data[metric][Idx], 
                label = 'CRONOS-AM'+" "+'(ours)',
                color = colors[solver], linewidth=2.5)
                plt.xlabel("Time (s)")

    if metric == 'train_loss':
        plt.ylabel("Training loss")
    
    elif metric == 'test_loss':
        plt.ylabel("Validation Loss")
    
    elif metric == 'train_acc':
        plt.ylabel("Training accuracy")
    
    else: 
        plt.ylabel("Validation Accuracy")

    plt.legend(title='Solver', loc = 'lower right', frameon=True)
    plt.savefig(dir, format='pdf')
    plt.close()

def plot_median_metric(solvers_metrics, metric, parameter, colors, dir, t_cutoff = None):
    if metric not in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
        raise ValueError("This metric is not supported!")
    alpha = 1
    for solver in solvers_metrics:
        data = solvers_metrics[solver]
        trajs = data[metric]
        grid_length, n_epochs = trajs.shape
        ql = np.quantile(trajs, 0.05, axis = 0)
        qu = np.quantile(trajs, 0.95, axis = 0)
        if metric in ['training_loss', 'test_loss']:
              if parameter == 'passes':
                if solver == 'Cronos_AM':
                  x = 2*np.arange(len(trajs[0]))
                  plt.semilogy(np.arange(n_epochs)+1, np.median(trajs, axis=0), label ='CRONOS-AM'+" "+'(ours)', color = colors[solver], alpha = alpha)
                  plt.xlabel("Data passes")
                else: 
                  x = np.arange(n_epochs)+1
                  plt.semilogy(np.arange(n_epochs)+1, np.median(trajs, axis=0), label = solver, color = colors[solver], alpha = alpha)
                  plt.xlabel("Data passes")
              elif parameter == 'time':
                if t_cutoff is not None:
                   x = np.median(np.cumsum(data['times'],axis = 1),axis=0)
                   J = np.where(x<=t_cutoff)
                   Idx = J[0]
                   x = x[Idx]
                   y = np.median(trajs,axis = 0)
                   y = y[Idx]
                else:
                   x = np.median(np.cumsum(data['times'],axis = 1),axis=0)
                   y = np.median(trajs, axis=0)
                if solver == 'Cronos_AM':
                  plt.plot(x, y, label = 'CRONOS-AM'+" "+'(ours)', color = colors[solver], alpha = alpha)
                else:
                  plt.plot(x, y, label = solver, color = colors[solver], alpha = alpha)
                plt.xlabel("Time(s)")
        else:
              if parameter == 'passes':
                if solver == 'Cronos_AM':
                  x = 2*np.arange(len(trajs[0]))
                  plt.plot(x, np.median(trajs, axis=0), label = 'CRONOS-AM'+" "+'(ours)', color = colors[solver], alpha = alpha)
                else: 
                  x = np.arange(n_epochs)+1
                  plt.plot(x, np.median(trajs, axis=0), label = solver, color = colors[solver], alpha = alpha)
                plt.xlabel("Data passes")
              elif parameter == 'time':
                  if t_cutoff is not None:
                     x = np.median(np.cumsum(data['times'],axis = 1),axis=0)
                     J = np.where(x<=t_cutoff)
                     Idx = J[0]
                     x = x[Idx]
                     y = np.median(trajs,axis = 0)
                     y = y[Idx]
                     ql = ql[Idx]
                     qu = qu[Idx]
                  else:
                    x = np.median(np.cumsum(data['times'],axis = 1),axis=0)
                    y = np.median(trajs, axis=0)
                  if solver == 'Cronos_AM':
                    plt.plot(x, y, label = 'CRONOS-AM'+" "+'(ours)', color = colors[solver], alpha = alpha)
                  else:
                    plt.plot(x, y, label = solver, color = colors[solver], alpha = alpha)
                  plt.xlabel("Time (s)")
        plt.fill_between(
            x,
            ql,
            qu,
            alpha=0.2*alpha,
            color=colors[solver],
            linewidth=0.0,
            rasterized=True)
    
    if metric == 'train_loss':
      plt.ylabel("Training loss")

    elif metric == 'test_loss':
         plt.ylabel("Validation Loss")

    elif metric == 'train_acc':
         plt.ylabel("Training accuracy")

    else:
      plt.ylabel("Validation Accuracy")

    plt.legend(title='Solver', loc = 'lower right', frameon=True)
    plt.savefig(dir, format='pdf')
    plt.close()

def sort_traj(solvers_metrics):
   for solver in solvers_metrics:
          if solver != 'Cronos_AM':
            indices = np.argsort(solvers_metrics[solver]['lr_grid'])
            solvers_metrics[solver]['lr_grid'] = solvers_metrics[solver]['lr_grid'][indices]
            solvers_metrics[solver]['test_acc'] = solvers_metrics[solver]['test_acc'][indices]
            solvers_metrics[solver]['times'] = solvers_metrics[solver]['times'][indices]
            l = len(solvers_metrics[solver]['lr_grid'])
            if l//2 !=0:
               m = np.int64((l-1)/2)
            else:
               j1 = np.int64(l/2)
               j2 = l-1
               traj1 =  solvers_metrics[solver]['test_acc'] = solvers_metrics[solver]['test_acc'][j1]
               traj2 = solvers_metrics[solver]['test_acc'] = solvers_metrics[solver]['test_acc'][j2]
               if np.max(traj1)>np.max(traj2):
                  m = j1
               else:
                  m = j2
            Idx = [0,m,l-1] 
            solvers_metrics[solver]['lr_grid'] = solvers_metrics[solver]['lr_grid'][Idx]
            solvers_metrics[solver]['test_acc'] = solvers_metrics[solver]['test_acc'][Idx]
            solvers_metrics[solver]['times'] = solvers_metrics[solver]['times'][Idx]  
   return solvers_metrics  

def multi_joy_plot(solvers_metrics, metric, colors, dir, t_cutoff = None):
   solvers_metrics = sort_traj(solvers_metrics)
   grid_length, n_epochs = solvers_metrics['Adam']['test_acc'].shape
   fig, axes = plt.subplots(grid_length, 1, sharex=True, figsize=(10, 8))
   # Set the y-axis label for the central subplot
   axes[1].set_ylabel('Validation accuracy', labelpad = 8)

   for i, ax in enumerate(axes):
        trajectories = {}
        for solver in solvers_metrics:
          if solver != 'Cronos_AM':
            trajectories[solver] = solvers_metrics[solver]['test_acc'][i]
          else:
            trajectories['Cronos_AM'] = solvers_metrics['Cronos_AM']['test_acc'][0]
        for  solver in trajectories:
          if solver == 'Cronos_AM':
             if metric == 'passes':
              x = 2*np.arange(len(solvers_metrics['Cronos_AM']['test_acc'][0]))
              y = trajectories[solver]
             elif metric == 'time':
                if t_cutoff is not None:
                   x = np.cumsum(solvers_metrics['Cronos_AM']['times'])
                   J = np.where(x<=t_cutoff)
                   Idx = J[0]
                   x = x[Idx]
                   y = trajectories[solver]
                   y = y[Idx]
                else:
                    x = np.cumsum(solvers_metrics['Cronos_AM']['times']) 
                    y = trajectories[solver]
             # else:
              #  x = np.cumsum(solvers_metrics['Cronos_AM']['times'])   
             ax.plot(x, y, color = colors[solver], label = 'CRONOS-AM'+" "+'(ours)')
          else:
            if metric == 'passes':
              x = np.arange(n_epochs)+1
              y = trajectories[solver] 
            elif metric == 'time':
              if t_cutoff is not None:
                   x = np.cumsum(solvers_metrics[solver]['times'][i])
                   J = np.where(x<=t_cutoff)
                   Idx = J[0]
                   x = x[Idx]
                   y = trajectories[solver]
                   y = y[Idx]
              else:
                x = np.cumsum(solvers_metrics[solver]['times'][i]) 
                y = trajectories[solver] 
            lr = solvers_metrics[solver]['lr_grid'][i]
            formatted_lr = f"{lr:.3e}"  # Format the learning rate in scientific notation with 6 digits
            label = f"{solver} (lr = {formatted_lr})"
            ax.plot(x, y, color=colors[solver], label = label)
        
        if metric == 'passes':
          u = n_epochs+1 
        elif metric == 'time':
          if t_cutoff is not None:
             u = t_cutoff
          else:
             u = np.maximum(np.max(np.cumsum(solvers_metrics['Cronos_AM']['times'])), np.max(np.cumsum(solvers_metrics['Shampoo']['times'][i])))  
        
        ax.set_xlim(0, u)  
          #ax.set_xlim(x.min(), x.max())
          #if i != len(trajectories) - 1:
              #ax.set_xticks([])  # Remove x-axis ticks for all but the last plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),frameon = 'true',fontsize=6)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon = 'true',fontsize=8.5)
       # ax.legend(loc = 'lower right', frameon = 'true',fontsize=6)
   if metric == 'passes':
    axes[-1].set_xlabel('Data passes')
   
   elif metric == 'time':
     axes[-1].set_xlabel('Time (s)') 
   
   # Adjust spacing between subplots
   plt.subplots_adjust(hspace=0.1) 
   fig.savefig(dir, format='pdf', bbox_inches='tight')
   plt.close()

def gpt2_benchmark_plot(results, dir):
    
    cronos_peak_test_acc = np.max(results['CRONOS']['val_acc'])
    grid_length, _ = results['AdamW']['test_acc'].shape

    for i in range(grid_length):
       t = np.cumsum(results['AdamW']['test_acc'][i], axis = 1)
       traj = results['AdamW']['test_acc'][i]
       plt.plot(t, traj, label = 'AdamW')
    
    plt.plot(cronos_peak_test_acc*np.ones(grid_length), label = 'CRONOS Peak Validation Accuracy')
    
    plt.xlabel("Time(s)")
    plt.ylabel("Validation Accuracy")


  




    