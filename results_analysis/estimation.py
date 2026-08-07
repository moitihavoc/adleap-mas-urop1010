import src.utils.plot as plt
import src.utils.stats as sts
import src.utils.results as rpck

import warnings
warnings.filterwarnings("ignore")

# 1. Defining analysis  settings
NEXP = 100
PATH = './results/'
SAVE = True

SHOW_SUMMARY = False
SHOW_PVALUE = False

PLOT =  True
PLOT_TYPE = 'lines'

ylabel = {
    'lines':{
        'reward':'Average Reward',
        'time':'Average Time (s)'},
    'cumlines':{
        'reward':'Cumulative Reward',
        'time':'Cumulative Time (s)'},
    'bars':{
        'reward':'Average Reward',
        'time':'Average Time (s)'},
    }       
envs = [
    #'LevelForagingEnvRAND_20x20_A5_T30',
    'LevelForagingEnvRAND_30x30_A5_T30',
        ]   
env_dict = {
    'LevelForagingEnv5':'Duo',
    'LevelForagingEnv6':'Strong',
    'LevelForagingEnv7':'Weak',
    'LevelForagingEnv8':'Wrong',
    'LevelForagingEnv9':'5Team',
}

methods_dict = {
    'pomcp':'POMCP',
    'ibpomcp':'IB-POMCP',
    'tbrhopomcp':'TB ρ-POMCP',
}
estimations_dict = {
    'pomce':'POMCE',
    #'aga':'AGA',
    #'abu':'ABU',
    #'oeate':'OEATE',
    #'bae':'BAE',
    #'oeate_a':'OEATA-A',
}
methods = [name for name in methods_dict]
estimations = [name for name in estimations_dict]

# Reading data
results = {}
for env in envs:
    results[env] = {}
    print('> Reading data for',env)
    for method in methods:
        if method == 'MT-MCTS':
            print('>>',methods_dict[method]+'_'+estimations_dict[estimation])
            results[env][methods_dict[method]+'_BAE'] = \
                rpck.read_estimation(nexp=NEXP,method=method,estimation=estimation,path=PATH,env=env)
        else:
            for estimation in estimations:
                print('>>',methods_dict[method]+'_'+estimations_dict[estimation])
                results[env][methods_dict[method]+'_'+estimations_dict[estimation]] = \
                    rpck.read_estimation(nexp=NEXP,method=method,estimation=estimation,path=PATH,env=env)
 
    # 2. Analysing via plot and pvalues
    if SHOW_SUMMARY:
        sts.summary(results=results,LaTeX=True)

    if SHOW_PVALUE:
        sts.pvalues(results=results,by_='iteration')

    if PLOT:
        for env in envs:
            print('|',env)
            for estimation in estimations:
                r = {}
                for name in results[env]:
                    method = name.split('_')[0]
                    r[method+'_'+estimations_dict[estimation]] = results[env][method+'_'+estimations_dict[estimation]]
                plt.type_estimation(r, estimation=estimation, env_name=env)
                plt.parameter_estimation(r, estimation=estimation,env_name=env)