import os
import sys
sys.path.append(os.getcwd())

import src.utils.plot as plt
import src.utils.stats as sts
import src.utils.results as rpck
# if warnings are disturbing the presentation, uncomment the lines bellow
import warnings
warnings.filterwarnings("ignore")

# 1. Defining analysis  settings
NEXP = 50
PATH = './results/ablation_q/'
QPARAM = ['q0','q01','q02','q03','q04','q05']
SAVE = True
OUTPUT_PATH = './plots/ablation_q/'

SHOW_SUMMARY = True
SHOW_PVALUE = True

PLOT = True
PLOT_TYPE = 'cumlines'

# select the target data
target_data = 'reward'
ylabel = {
    'lines':{
        'reward':'Average Reward',
        'time':'Average Time (s)'},
    'cumlines':{
        'reward':'Cumulative Reward',
        'time':'Cumulative Time (s)',
            'coop':'Together Accomp.'},
        'bars':{
        'reward':'Average Reward',
        'time':'Average Time (s)',
        'cost':'Time to Accomp. (s)'},
    }    

# select the target environments
ENVS = {
    #'TigerEnv'         :[0],
    'MazeEnv'          :[1],
    'RockSampleEnv'    :[3],
    'LevelForagingEnv' :[1,3],
    #'TagEnv'           :[0],
    #'LaserTagEnv'      :[0]
}

# select the target methods
methods = ['ibpomcp']
methods_dict = {
    'ibpomcpq0'       :'IB-POMCP (q=0.0)',
    'ibpomcpq01'       :'IB-POMCP (q=0.1)',
    'ibpomcpq02'       :'IB-POMCP (q=0.2)',
    'ibpomcpq03'       :'IB-POMCP (q=0.3)',
    'ibpomcpq04'       :'IB-POMCP (q=0.4)',
    'ibpomcpq05'       :'IB-POMCP (q=0.5)',
}

all_results = {}
for e in ENVS:
    for s in ENVS[e]:
        env = e+str(s)
        results = {}
        for q in QPARAM:
            path = PATH+q+'/'
            for method in methods:
                method_q = method+q
                results[methods_dict[method_q]] = rpck.read(nexp=NEXP,method=method,path=path,env=env)
        all_results[env] = results

import matplotlib.pyplot as plt
import src.utils.stats as sts
from src.utils.plot import *

COLOR_DICT['IB-POMCP (q=0.0)']  = '#08519c'
COLOR_DICT['IB-POMCP (q=0.1)'] = '#3182bd' 
COLOR_DICT['IB-POMCP (q=0.2)'] = '#41ab5d'
COLOR_DICT['IB-POMCP (q=0.3)'] = '#fec44f' 
COLOR_DICT['IB-POMCP (q=0.4)'] = '#fe9929'
COLOR_DICT['IB-POMCP (q=0.5)'] = '#de2d26'

MARKER_DICT['IB-POMCP (q=0.0)']  = '.'
MARKER_DICT['IB-POMCP (q=0.1)'] = 'o'
MARKER_DICT['IB-POMCP (q=0.2)'] = 's'
MARKER_DICT['IB-POMCP (q=0.3)'] = '^'
MARKER_DICT['IB-POMCP (q=0.4)'] = 'p'
MARKER_DICT['IB-POMCP (q=0.5)'] = 'h'

LINESTYLE_VEC_DICT['IB-POMCP (q=0.0)']  = '-'
LINESTYLE_VEC_DICT['IB-POMCP (q=0.1)'] = '--'
LINESTYLE_VEC_DICT['IB-POMCP (q=0.2)'] = ':'
LINESTYLE_VEC_DICT['IB-POMCP (q=0.3)'] = '-.'
LINESTYLE_VEC_DICT['IB-POMCP (q=0.4)'] = (0, (3, 5, 1, 5))         
LINESTYLE_VEC_DICT['IB-POMCP (q=0.5)'] = (0, (3, 5, 1, 5, 1, 5))

target = 'reward'

FIG_COUNTER = 0
for e in ENVS:
    for s in ENVS[e]:
        env = e+str(s)
        print('\n>',env)
        results = all_results[env]
        sts.summary(results=results,target_data=target,cumsum=True,LaTeX=True)
        sts.pvalues(results=results,target_data=target,cumsum=True,by_='experiment')
        plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

        y = {}
        y_lower = {}
        y_upper = {}
        counter = 0
        for method in results:
            print('>',method)
            data = [exp[target_data] for exp in results[method]]
            y[method], y_lower[method], y_upper[method] =\
                sts.by_iteration(data,complete_with='zero',cumsum=True,fixed_max_len=200)
            
            plt.fill_between(range(len(y_lower[method])),y_lower[method],y_upper[method],color=COLOR_DICT[method],alpha=0.4)
            plt.plot(y[method],label=method,
                color=COLOR_DICT[method],marker=MARKER_DICT[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC_DICT[method], markeredgecolor='black')
            counter += 1
            
        plt.legend(loc='best',ncol=1,fontsize=18,edgecolor='black')

        plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
        plt.xlabel('Iterations',fontdict=FONT_DICT)
        plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
        plt.ylabel('Cumulative Reward',fontdict=FONT_DICT)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH+env+'_q_summary_'+target+'.pdf')

        if FIG_COUNTER == 0:       
            FIG_COUNTER += 1     
            plot_only_legend(results, save=True, savepath=OUTPUT_PATH, env_name='',ncol=3)
        FIG_COUNTER += 1

