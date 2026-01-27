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
ENVS = [
    #'TigerEnv',
    #'MazeEnv',
    #'RockSampleEnv',
    'LevelForagingEnv',
    #'TagEnv',
    #'LaserTagEnv'
]
SCENARIOS = [5,6,7,8,9,10]
envs = [e+str(s) for s in SCENARIOS for e in ENVS]

# select the target methods
methods_dict = {
    'pomcp':'POMCP',
    #'prpomcp':'PR-POMCP',
    #'iucbpomcp':'IUCB-POMCP',
    'ibpomcp':'IB-POMCP',
    'tbrhopomcp':'TB ρ-POMCP',
    'rhopomcp':'ρ-POMCP',
    #'mcts_bae':'BAE',
    #'mcts_aga':'AGA',
    #'mcts_abu':'ABU',
    #'mcts_oeate_a':'OEATE-A',
}
methods = [name for name in methods_dict]

all_results = {}
for env in envs:
    print('\n>',env)

    results = {}
    for method in methods:
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
                    xlabel='Iteration',save=SAVE,savepath='./plots/',env_name=env)

            elif PLOT_TYPE == 'cumlines':
                plt.cumlines(results=results,target_data=td,
                            ylabel=ylabel[PLOT_TYPE][td],xlabel='Iteration',
                            save=SAVE,savepath='./plots/',env_name=env)
            elif PLOT_TYPE == 'bars':
                plt.bars(results=results,target_data=td,ylabel=ylabel[PLOT_TYPE][td],
                            save=SAVE,savepath='./plots/',env_name=env)
            else:
                print('Plot type',PLOT_TYPE,'is not available.')

if PLOT and PLOT_TYPE=='summarized_bar_plot':
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