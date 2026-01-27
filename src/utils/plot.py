import matplotlib.pyplot as plt
import numpy as np
import os
import src.utils.stats as sts

NEXP = 50

FIG_COUNTER = 0
FIGSIZE = (6.4,5.4)

FONTSIZE = 28
LEGEND_FONTSIZE = 12
FONT_DICT = {
        'weight': 'bold',
        'size': 26,
        }
TICK_FONTSIZE = 20

COLOR_VEC   = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple','tab:brown','tab:pink','tab:olive']
            
MARKER_SIZE = 18
MARK_EVERY = 20
MARKER_VEC = ['o','^','p','s','X','o']

LINEWIDTH = 5
LINESTYLE_VEC = ['--','-',':','-.','-.']

COLOR_DICT = {
    'pomcp':'tab:blue',
    'POMCP':'tab:blue',
    'ibpomcp':'tab:orange',
    'IB-POMCP':'tab:orange',
    'tbrhopomcp':'tab:purple',
    'TB ρ-POMCP':'tab:purple',
    'rhopomcp':'tab:brown',
    'ρ-POMCP':'tab:brown',
}
MARKER_DICT = {
    'pomcp':'o',
    'POMCP':'o',
    'ibpomcp':'^',
    'IB-POMCP':'^',
    'rhopomcp':'p',
    'ρ-POMCP':'p',
    'tbrhopomcp':'s',
    'TB ρ-POMCP':'s',
}
LINESTYLE_VEC_DICT = {
    'pomcp':'--',
    'POMCP':'--',
    'ibpomcp':'-',
    'IB-POMCP':'-',
    'rhopomcp':':',
    'ρ-POMCP':':',
    'tbrhopomcp':'-.',
    'TB ρ-POMCP':'-.',
}

def lines(results,target_data,ylabel='y-axis',xlabel='x-axis',save=False,savepath='./plots/',env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    for method in results:
        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(results[method],target_data=target_data,complete_with='zero')
        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR_DICT[method],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR_DICT[method],marker=MARKER_DICT[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC_DICT[method], markeredgecolor='black')
        counter += 1
    plt.legend(loc='best',ncol=1,fontsize=LEGEND_FONTSIZE,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    plt.savefig(savepath+target_data+'_lines.pdf')
    plt.show()
    FIG_COUNTER += 1

def cumlines(results,target_data,ylabel='y-axis',xlabel='x-axis',save=False,savepath='./plots/',env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    for method in results:

        data = [exp[target_data] for exp in results[method]]
        for exp in range(len(data)):
            for i in range(len(data[exp])):
                data[exp][i] = np.mean(data[exp][i])

        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(data,complete_with='zero',cumsum=True,fixed_max_len=200)
        
        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR_DICT[method],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR_DICT[method],marker=MARKER_DICT[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC_DICT[method], markeredgecolor='black')
        counter += 1
        
    if env_name == 'LevelForagingEnv6':
        plt.legend(loc='best',ncol=1,fontsize=18,edgecolor='black')

    if env_name == 'LevelForagingEnv8' or env_name == 'LevelForagingEnv9' or env_name == 'LevelForagingEnv10':
        plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)

    if env_name == 'LevelForagingEnv5' or env_name == 'LevelForagingEnv8':
        plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    b, t = plt.ylim()
    plt.ylim(0,t)
    plt.tight_layout()

    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+env_name+'_'+target_data+'_cumlines.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1

def bars(results,target_data,ylabel='y-axis',save=False,savepath='./plots/',env_name=''):
    global FIG_COUNTER, FIGSIZE
    #plt.figure(num=FIG_COUNTER,figsize=(FIGSIZE))

    xlabels = []
    heights = []
    errors = []
    for method in results:
        if target_data == 'cost':
            r = []
            for exp in results[method]:
                time = 0
                last_time = 0
                total_reward = 0
                exptime2task = []
                for i in range(len(exp['reward'])):
                    time += exp['time'][i]
                    if exp['reward'][i] != 0:
                        total_reward += exp['reward'][i]
                        for _ in range(int(exp['reward'][i])):
                            exptime2task.append(time - last_time)
                            last_time = time
                        time = 0

                for _ in range(int(19-total_reward)):
                    exptime2task.append(sum(exp['time']))

                r.append(np.mean(exptime2task))
            m, l, u = sts.mean_confidence_interval(r,by_='experiment',confidence=0.99)
            heights.append(m)
            errors.append((u-l)/2)
        else:
            m, l, u =\
                sts.by_experiment(results[method],target_data=target_data)
            heights.append(np.mean(m))
            errors.append(np.mean(u-l)/2)
        xlabels.append(method)

    if env_name == 'LevelForagingEnv5' or  env_name == 'LevelForagingEnv6' or  env_name == 'LevelForagingEnv7':
        plt.figure(num=FIG_COUNTER,figsize=(6.4,4.6))
        xlabels = ['' for n in xlabels]
    else:
        plt.figure(num=FIG_COUNTER,figsize=(FIGSIZE))

    colors = [COLOR_DICT[method] for method in results]
    plt.bar(range(len(heights)),heights,width=0.8,align='center',color=colors,edgecolor='black',
                linewidth=1, tick_label=xlabels, yerr=errors,capsize=5)
    plt.xticks(fontsize=TICK_FONTSIZE,fontweight='bold',rotation=30)

    if env_name == 'LevelForagingEnv5' or env_name == 'LevelForagingEnv8':
        plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    
    plt.tight_layout()
    
    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+env_name+'_'+target_data+'_bars.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1

def type_estimation(results, estimation,\
    ylabel='Type Estimation Err',xlabel='Iteration',\
    save=True, savepath='./plots/', env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    print('|| type estimation')
    for method in results:
        y[method], y_lower[method], y_upper[method] = [], [], []
        
        data = [exp['typeestimation_err'] for exp in results[method]]
        for exp in range(len(data)):
            for i in range(len(data[exp])):
                data[exp][i] = np.mean(data[exp][i])

        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(data, complete_with='last', fixed_max_len=200)

        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR_VEC[counter%len(COLOR_VEC)],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR_VEC[counter%len(COLOR_VEC)],marker=MARKER_VEC[counter%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[counter%len(LINESTYLE_VEC)], markeredgecolor='black')
        counter += 1
    plt.legend(loc='best',ncol=1,fontsize=LEGEND_FONTSIZE,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+env_name+'_'+estimation+'_type_lines.pdf')
    plt.show()
    FIG_COUNTER += 1

def parameter_estimation(results, estimation, nparams=3,\
    ylabel='Parameter Estimation',xlabel='Iteration',\
    save=True, savepath='./plots/', env_name=''):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    print('|| parameter estimation')
    for method in results:
        y[method], y_lower[method], y_upper[method] = [], [], []

        data = [exp['parameterestimation_err'] for exp in results[method]]
        for exp in range(len(data)):
            for i in range(len(data[exp])):
                data[exp][i] = np.mean(data[exp][i])

        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(data, complete_with='last', fixed_max_len=200)

        plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR_VEC[counter%len(COLOR_VEC)],alpha=0.4)
        plt.plot(y[method],label=method,
            color=COLOR_VEC[counter%len(COLOR_VEC)],marker=MARKER_VEC[counter%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[counter%len(LINESTYLE_VEC)], markeredgecolor='black')
        counter += 1
    plt.legend(loc='best',ncol=1,fontsize=LEGEND_FONTSIZE,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+env_name+'_'+estimation+'_parameter_lines.pdf')
    plt.show()
    FIG_COUNTER += 1


def adversarial_analysis(results):
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.rc('text',usetex=True)
    ###
    ### ADVERSARY DETECTION
    ###
    FIG_COUNTER = 0
    FIGSIZE = (6.4,5.4)
    for env in results:
        approaches_line = {}
        for approach_name in results[env]:
            print('\n',env,approach_name)
            # Analysing via plot and pvalues
            plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)
            n_agents = len(results[env][approach_name][0]['typeestimation'][0])

            if approach_name == 'MCTS_BAE' or approach_name == 'MCTS_AGA-BAE' or approach_name == 'MCTS_ABU-BAE' or approach_name == 'MCTS_OEATA-A':
                for ag in range(n_agents):
                    # 1. Formating data
                    line = []
                    for exp in range(NEXP):
                        line.append(np.zeros(200))
                        for it in range(200):
                            if len(results[env][approach_name][exp]['typeestimation']) > it:
                                line[exp][it] += (results[env][approach_name][exp]['typeestimation'][it][ag][-1])
                            else:
                                line[exp][it] += (results[env][approach_name][exp]['typeestimation'][-1][ag][-1])
                            
                            line[exp][it] = 1.0 if line[exp][it] > 1.0 else line[exp][it]
                            line[exp][it] = 0.0 if line[exp][it] < 0.0 else line[exp][it]

                    # 2. Summarising data into a line (and confidence interval)
                    data = sts.mean_confidence_interval(line)
                    if ag == n_agents-1:
                        print('mean %.2f \pm %.2f' %(np.mean(data[0]), np.mean(data[2][1:] - data[1][1:])/2))
                    approaches_line[approach_name] = data[0]

                    # 3. Plotting
                    if ag+1 != n_agents and approach_name:
                        plt.plot(data[0],label=r"""Agent $\omega_"""+str(ag+1)+r"""$""",
                            color=COLOR_VEC[ag%len(COLOR_VEC)],marker=MARKER_VEC[ag%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[ag%len(LINESTYLE_VEC)], markeredgecolor='black')
                    else:
                        if env == 'LevelForagingEnv5' or env == 'LevelForagingEnv6':
                            ag += 2
                        plt.plot(data[0],label=r"""Agent $\omega_"""+str(ag+1)+r"""$\\(Impostor $\psi$)""",
                            color=COLOR_VEC[ag%len(COLOR_VEC)],marker=MARKER_VEC[ag%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[ag%len(LINESTYLE_VEC)], markeredgecolor='black')
                    plt.fill_between(range(len(data[1])),data[1],data[2],color=COLOR_VEC[ag%len(COLOR_VEC)],alpha=0.4)
            else:
                for ag in range(n_agents):
                    # 1. Formating data
                    line = []
                    for exp in range(NEXP):
                        line.append(np.zeros(200))
                        for it in range(200):
                            if len(results[env][approach_name][exp]['typeestimation']) > it:
                                line[exp][it] += (results[env][approach_name][exp]['typeestimation'][it][-1][ag])
                            else:
                                line[exp][it] += (results[env][approach_name][exp]['typeestimation'][-1][-1][ag])
                            
                            line[exp][it] = 1.0 if line[exp][it] > 1.0 else line[exp][it]
                            line[exp][it] = 0.0 if line[exp][it] < 0.0 else line[exp][it]

                    # 2. Summarising data into a line (and confidence interval)
                    data = sts.mean_confidence_interval(line)
                    if ag == n_agents-1:
                        print('mean %.2f \pm %.2f' %(np.mean(data[0]), np.mean(data[2][1:] - data[1][1:])/2))
                    approaches_line[approach_name] = data[0]

                    # 3. Plotting
                    if ag+1 != n_agents and approach_name:
                        plt.plot(data[0],label=r"""Agent $\omega_"""+str(ag+1)+r"""$""",
                            color=COLOR_VEC[ag%len(COLOR_VEC)],marker=MARKER_VEC[ag%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[ag%len(LINESTYLE_VEC)], markeredgecolor='black')
                    else:
                        if env == 'LevelForagingEnv5' or env == 'LevelForagingEnv6':
                            ag += 2
                        plt.plot(data[0],label=r"""Agent $\omega_"""+str(ag+1)+r"""$\\(Impostor $\psi$)""",
                            color=COLOR_VEC[ag%len(COLOR_VEC)],marker=MARKER_VEC[ag%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[ag%len(LINESTYLE_VEC)], markeredgecolor='black')
                    plt.fill_between(range(len(data[1])),data[1],data[2],color=COLOR_VEC[ag%len(COLOR_VEC)],alpha=0.4)
                
            plt.hlines([1/n_agents for n in range(200)],0,200,colors='black',LINESTYLE_VECs=':',linewidth=2)
            #plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center',ncol=2,fontsize=LEGEND_FONTSIZE+10,edgecolor='black')
            if env == 'LevelForagingEnv7' or env == 'LevelForagingEnv8':
                plt.xlabel('Iteration',fontdict=FONT_DICT)
            plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
            if env == 'LevelForagingEnv5' or env == 'LevelForagingEnv7':
                plt.ylabel(r'$P(\omega_\psi = \psi)$',fontdict=FONT_DICT)
            if env == 'LevelForagingEnv5' or env == 'LevelForagingEnv6':
                plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=TICK_FONTSIZE,rotation=45)
            else:
                plt.yticks([0.15,0.25,0.35,0.45,0.55],fontsize=TICK_FONTSIZE,rotation=45)
            plt.tight_layout()
            plt.savefig('plots/'+str(env)+'_'+str(approach_name)+'_lines.pdf')
            FIG_COUNTER += 1

        from scipy.stats import ttest_ind
        pvalues = {}
        for method1 in approaches_line:
            for method2 in approaches_line:
                pvalues[(method1,method2)] = ttest_ind(approaches_line[method1],approaches_line[method2],equal_var=False)[1]

        print('\n',[method for method in approaches_line])
        for method1 in approaches_line:
            for method2 in approaches_line:
                print('%.2f' % (pvalues[(method1,method2)]) + '\t',end='')
            print()

    for env in results:
        approaches_line == {}
        for approach_name in results[env]:
            print('\n',env,approach_name)

            n_agents = len(results[env][approach_name][0]['typeestimation'][0])

            for ag in range(n_agents):
                # 1. Formating data
                line = []
                for exp in range(NEXP):
                    line.append(np.zeros(200))
                    for it in range(200):
                        if len(results[env][approach_name][exp]['time']) > it:
                            line[exp][it] += (results[env][approach_name][exp]['time'][it])
                        else:
                            line[exp][it] += (results[env][approach_name][exp]['time'][-1])

                # 2. Summarising data into a line (and confidence interval)
                data = sts.mean_confidence_interval(line)
                if ag == n_agents-1:
                    print('mean %.2f \pm %.2f' %(np.mean(data[0]), np.mean(data[2][1:] - data[1][1:])/2))
                approaches_line[approach_name] = data[0]

        from scipy.stats import ttest_ind
        pvalues = {}
        for method1 in approaches_line:
            for method2 in approaches_line:
                pvalues[(method1,method2)] = ttest_ind(approaches_line[method1],approaches_line[method2],equal_var=False)[1]

        print('\n',[method for method in approaches_line])
        for method1 in approaches_line:
            for method2 in approaches_line:
                print('%.2f' % (pvalues[(method1,method2)]) + '\t',end='')
            print()
                    
def adversarial_estimation(results):
    FIG_COUNTER = 0
    FIGSIZE = (6.4,5.4)
    for env in results:
        plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)
        counter = 0
        for approach_name in results[env]:
            n_agents = len(results[env][approach_name][0]['typeestimation'][0])
            # 1. Formating data
            line = []
            for exp in range(NEXP):
                line.append(np.zeros(200))
                for it in range(200):
                    if len(results[env][approach_name][exp]['typeestimation']) > it:
                        line[exp][it] += (results[env][approach_name][exp]['typeestimation'][it][n_agents-1][-1])
                    else:
                        line[exp][it] += (results[env][approach_name][exp]['typeestimation'][-1][n_agents-1][-1])
                    
                    line[exp][it] = 1.0 if line[exp][it] > 1.0 else line[exp][it]
                    line[exp][it] = 0.0 if line[exp][it] < 0.0 else line[exp][it]

            # 2. Summarising data into a line (and confidence interval)
            data = sts.mean_confidence_interval(line)
            plt.plot(data[0],label=approach_name,
                color=COLOR_VEC[counter%len(COLOR_VEC)],marker=MARKER_VEC[counter%len(MARKER_VEC)], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC[counter%len(LINESTYLE_VEC)], markeredgecolor='black')
            plt.fill_between(range(len(data[1])),data[1],data[2],color=COLOR_VEC[counter%len(COLOR_VEC)],alpha=0.4)
            counter+=1
        plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center',ncol=2,fontsize=LEGEND_FONTSIZE+10,edgecolor='black')
        plt.xlabel('Iteration',fontdict=FONT_DICT)
        plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
        plt.ylabel('Impostor Prob. (%)',fontdict=FONT_DICT)
        plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
        plt.tight_layout()
        plt.savefig('plots/'+str(env)+'_lines.pdf')
        FIG_COUNTER += 1