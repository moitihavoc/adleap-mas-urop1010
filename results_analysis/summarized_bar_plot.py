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
SAVE = True
OUTPUT_PATH = './plots/'

SHOW_SUMMARY = True
SHOW_PVALUE = True

PLOT = True
PLOT_TYPE = 'cumlines'

# select the target data
target_data = ['reward'] 
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
    #'MazeEnv'          :[0,1,2,3],
    #'RockSampleEnv'    :[0,1,2,3],
    #'LevelForagingEnv' :[0,1,2,3,4],
    #'TagEnv'           :[0],
    #'LaserTagEnv'      :[0]
}

# select the target methods
methods_dict = {
    'pomcp'         :'POMCP',
    'tbrhopomcp'    :'TB ρ-POMCP',
    #'iprpomcp'     :'IPR-POMCP',
    #'iucbpomcp'    :'IUCB-POMCP',
    'ibpomcp'       :'IB-POMCP',
    'rhopomcp'      :'ρ-POMCP',
    'ipftreed'      :'IPFT'
}
methods = [name for name in methods_dict]

all_results = {}
for e in ENVS:
    for s in ENVS[e]:
        env = e+str(s)
        print('\n>',env)

        results = {}
        for method in methods:
            if method == 'rhopomcp' or method == 'ipftreed':
                results[methods_dict[method]] = \
                    rpck.read(nexp=NEXP,method=method,path='./results/10sec/',env=env)
            else:
                results[methods_dict[method]] = \
                    rpck.read(nexp=NEXP,method=method,path=PATH,env=env)
        all_results[env] = results

import matplotlib.pyplot as plt
import numpy as np
import src.utils.stats as sts

target = 'reward'
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
LINESTYLE_DICT = {
    'pomcp':'--',
    'POMCP':'--',
    'ibpomcp':'-',
    'IB-POMCP':'-',
    'rhopomcp':':',
    'ρ-POMCP':':',
    'tbrhopomcp':'-.',
    'TB ρ-POMCP':'-.',
}

plt.figure(figsize=FIGSIZE)

heights = []
errors = []
for env in all_results:  
    for method in all_results[env]:
        r = [np.mean(
            #np.cumsum(all_results[env][method][i][target])
            all_results[env][method][i][target]
                ) for i in range(len(all_results[env][method]))]
        heights.append(np.mean(r))
        errors.append(np.std(r))

xpos = [ 0.4, 0.8, 1.2, 1.6,\
            2.4, 2.8, 3.2, 3.6,\
            4.4, 4.8, 5.2, 5.6,\
            6.4, 6.8, 7.2, 7.6,\
            8.4, 8.8, 9.2, 9.6,\
        10.4,10.8,11.2,11.6]
xlabels = [1,3,5,7,9,11]
colors = [COLOR_DICT[method] for method in results]
plt.bar(xpos,heights,width=0.4,align='center',color=colors,edgecolor='black',
            linewidth=1, yerr=errors,capsize=5)

plt.xlabel('Number of Agents',fontdict=FONT_DICT)
plt.xticks(xlabels,xlabels,fontsize=TICK_FONTSIZE,rotation=45)

plt.ylabel('Average Time (s)',fontdict=FONT_DICT)
plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)

plt.tight_layout()
plt.savefig('agents_'+target+'_bars.pdf')
FIG_COUNTER += 1