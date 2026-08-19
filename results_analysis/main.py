
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
TIME_BUDGET = '1sec'

PATH = './results/' + TIME_BUDGET + '/'
OUTPUT_PATH = './plots/' + TIME_BUDGET + '/'

SAVE = True
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
    'MazeEnv'          :[0,1,2,3],
    'RockSampleEnv'    :[0,1,2,3],
    'LevelForagingEnv' :[0,1,2,3,4],
    'TagEnv'           :[0],
    'LaserTagEnv'      :[0]
}

# select the target methods
methods_dict = {
    'pomcp'         :'POMCP',
    'tbrhopomcp'    :'TB ρ-POMCP',
    #'iprpomcp'      :'IPR-POMCP',
    #'iucbpomcp'     :'IUCB-POMCP',
    #'libpomcp'      :'LIB-POMCP',
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
            
        # 2. Analysing via plot and pvalues
        for td in target_data:
            if SHOW_SUMMARY:
                if PLOT_TYPE =='cumlines':
                    sts.summary(results=results,target_data=td,cumsum=True,LaTeX=True)
                else:
                    sts.summary(results=results,target_data=td,cumsum=False,LaTeX=True)

            if SHOW_PVALUE:
                if PLOT_TYPE =='cumlines':
                    sts.pvalues(results=results,target_data=td,cumsum=True,by_='experiment')
                else:
                    sts.pvalues(results=results,target_data=td,cumsum=False,by_='experiment')

            if PLOT:
                if PLOT_TYPE == 'lines':
                    plt.lines(results=results,target_data=td,ylabel=ylabel[PLOT_TYPE][td],
                        xlabel='Iteration',save=SAVE,savepath=OUTPUT_PATH,env_name=env)

                elif PLOT_TYPE == 'cumlines':
                    plt.cumlines(results=results,target_data=td,
                                ylabel=ylabel[PLOT_TYPE][td],xlabel='Iteration',
                                save=SAVE,savepath=OUTPUT_PATH,env_name=env)
                elif PLOT_TYPE == 'bars':
                    plt.bars(results=results,target_data=td,ylabel=ylabel[PLOT_TYPE][td],
                                save=SAVE,savepath=OUTPUT_PATH,env_name=env)
                else:
                    print('Plot type',PLOT_TYPE,'is not available.')
