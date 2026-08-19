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
PATH = './results/'
TIME_BUDGETS = ['1sec','2sec','3sec']

SAVE = True
OUTPUT_PATH = './plots/time_budgets/'

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
    'TigerEnv'         :[0],
    'MazeEnv'          :[0,1,2,3],
    'RockSampleEnv'    :[0,1,2,3],
    'LevelForagingEnv' :[0,1,2,3,4],
    'TagEnv'           :[0],
    'LaserTagEnv'      :[0]
}

# select the target methods
methods_dict = {#
    'pomcp'         :'POMCP',
    'tbrhopomcp'    :'TB ρ-POMCP',
    #'iprpomcp'     :'IPR-POMCP',
    #'iucbpomcp'    :'IUCB-POMCP',
    'ibpomcp'       :'IB-POMCP',
    #'rhopomcp'      :'ρ-POMCP',
    #'ipftreed'      :'IPFT'
}
methods = [name for name in methods_dict]

all_results = {}
for e in ENVS:
    for s in ENVS[e]:
        env = e+str(s)
        print(env)
        results = {}
        for t in TIME_BUDGETS:
            path = PATH+t+'/'
            for method in methods:
                if method not in results:
                    results[method] = [\
                        rpck.read(nexp=NEXP,method=method,path=path,env=env)
                    ]
                else:
                    results[method].append(\
                        rpck.read(nexp=NEXP,method=method,path=path,env=env)
                    )
        all_results[env] = results


import matplotlib.pyplot as plt
import src.utils.stats as sts
from src.utils.plot import COLOR_DICT, MARKER_DICT, MARKER_SIZE, LINEWIDTH, LINESTYLE_VEC_DICT, FONT_DICT, FIGSIZE, TICK_FONTSIZE

MARK_EVERY = 1

target = 'reward'

FIG_COUNTER = 0
for e in ENVS:
    for s in ENVS[e]:
        plt.figure(FIG_COUNTER,figsize=FIGSIZE)
        x = [i for i in range(len(TIME_BUDGETS))]
        y = [[] for _ in range(len(methods_dict))]
        y_l = [[] for _ in range(len(methods_dict))]
        y_u = [[] for _ in range(len(methods_dict))]

        env = e+str(s)
        results = all_results[env]
        for i in range(len(results)):
            method = methods_dict[methods[i]]
            for j in range(len(x)):
                m, l, u = sts.by_experiment(results=results[methods[i]][j], target_data=target_data, cumsum=True, fixed_max_len=200)
                m, l, u = sts.mean_confidence_interval(m, by_='iteration')

                y[i].append(m)
                y_l[i].append(l)
                y_u[i].append(u)
            plt.fill_between(range(len(y_l[i])),y_l[i],y_u[i],color=COLOR_DICT[method],alpha=0.4)
            plt.plot(y[i],label=method,
                color=COLOR_DICT[method],marker=MARKER_DICT[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,
                linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC_DICT[method], markeredgecolor='black')

        plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
        plt.xticks(x,fontsize=TICK_FONTSIZE,rotation=45)
        plt.xlabel('Time Budget (s)',fontdict=FONT_DICT)
        plt.ylabel('Cumulative Reward',fontdict=FONT_DICT)

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH+env+'_tb_summary_'+target+'.pdf')
        FIG_COUNTER += 1