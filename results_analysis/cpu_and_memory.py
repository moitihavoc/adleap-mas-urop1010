import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import os

NEXP = 50
ENVS_DICT = {'tiger0':'Tiger (T0)','maze2':'MazeDots (M2)',
             'rocksample2':'RockSample44 (R2)','levelforaging4':'TheOffice (F4)'}
ENVS = ['tiger0','maze2','rocksample2','levelforaging4']
METHODS = ['pomcp','ibpomcp','tbrhopomcp','rhopomcp']
METHODS_DICT = {
    'pomcp':'POMCP',
    'ibpomcp':'IB-POMCP',
    'tbrhopomcp':r"TB-$\rho$-POMCP",
    'rhopomcp':r"$\rho$-POMCP"
}
COLORS = {
    'pomcp':'tab:blue',
    'POMCP':'tab:blue',

    'ibpomcp':'tab:orange',
    'IB-POMCP':'tab:orange',

    'tbrhopomcp':'tab:purple',
    'TB ρ-POMCP':'tab:purple',

    'rhopomcp':'tab:brown',
    'ρ-POMCP':'tab:brown',

}
MARKER = {
    'pomcp':'o',
    'POMCP':'o',

    'ibpomcp':'^',
    'IB-POMCP':'^',

    'rhopomcp':'p',
    'ρ-POMCP':'p',

    'tbrhopomcp':'s',
    'TB ρ-POMCP':'s',
}
LINESTYLE = {
    'pomcp':'--',
    'POMCP':'--',

    'ibpomcp':'-',
    'IB-POMCP':'-',

    'rhopomcp':':',
    'ρ-POMCP':':',

    'tbrhopomcp':'-.',
    'TB ρ-POMCP':'-.',
}

FIG_COUNTER = 0
FIGSIZE = (6.4,5.4)
FONTSIZE = 28
LEGEND_FONTSIZE = 18
FONT_DICT = {
        'weight': 'bold',
        'size': 26,
        }
TICK_FONTSIZE = 20
MARKER_SIZE = 18
MARK_EVERY = 100
LINEWIDTH = 5

def mean_confidence_interval(data, confidence=0.95):
    mean = data.mean(axis=1)
    ci = data.apply(lambda x: stats.t.interval(confidence, len(x)-1, loc=np.mean(x), scale=stats.sem(x)), axis=1)
    lo = [ci[i][0] for i in range(len(ci))]
    hi = [ci[i][1] for i in range(len(ci))]
    return mean, lo, hi

def plot_lines(results,mean_label,low_label,high_label,xlabel,ylabel,figname):
    global FIG_COUNTER, FIGSIZE, METHODS_DICT, METHODS, FONT_DICT, TICK_FONTSIZE, COLORS

    FIG_COUNTER += 1
    plt.figure(FIG_COUNTER,figsize=FIGSIZE)
    for method in METHODS:
        if method in results:
            x = [i for i in range(1,len(results[method][mean_label]))]
            y = results[method][mean_label][1:]
            y_bci = results[method][low_label][1:]
            y_hci = results[method][high_label][1:]
            plt.plot(x,y,label=METHODS_DICT[method],
                color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
            plt.fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,fontweight='bold',rotation=45)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.legend(loc='lower right',fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(figname)

def plot_cumlines(results,mean_label,low_label,high_label,xlabel,ylabel,figname):
    global FIG_COUNTER, FIGSIZE, METHODS_DICT, METHODS, FONT_DICT, TICK_FONTSIZE, COLORS

    FIG_COUNTER += 1
    plt.figure(FIG_COUNTER,figsize=FIGSIZE)
    for method in METHODS:
        if method in results:
            x = [i for i in range(len(results[method][mean_label]))]
            y = np.cumsum(results[method][mean_label])
            y_bci = (y+np.cumsum(results[method][low_label]-results[method][mean_label]))
            y_hci = (y+np.cumsum(results[method][high_label]-results[method][mean_label]))
            plt.plot(x,y,label=METHODS_DICT[method],
                color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
            plt.fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.legend(loc='lower right',fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(figname)

def plot_box(results,data_label,ylabel,figname):
    global FIG_COUNTER, FIGSIZE, METHODS_DICT, METHODS, FONT_DICT, TICK_FONTSIZE, COLORS

    FIG_COUNTER += 1
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = []
    for method in METHODS:
        data.append(results[method][data_label])

    boxprops = dict(linestyle='--', linewidth=1.5, color='black')
    flierprops = dict(marker='o', markerfacecolor='tab:cyan', markersize=5, alpha=0.5)
    medianprops = dict(linestyle='-.', linewidth=1.5, color='tab:blue')
    whiskerprops = dict(linestyle='-', linewidth=1.5, color='tab:green')
    ax.boxplot(data, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops)

    # Personalizar os títulos e os rótulos dos eixos
    ax.set_ylabel(ylabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)

    # Personalizar o eixo x
    xticks = METHODS_DICT.values()
    ax.set_xticklabels(xticks)
    plt.tight_layout()
    plt.savefig(figname)
















print('Reading')
results = {}
for env in ENVS:
    print('-',env)
    results[env] = {}
    for method in METHODS:
        if os.path.exists('./cpu_mem_profile/pickles/'+env+'_'+method+'.pickle'):
            with open('./cpu_mem_profile/pickles/'+env+'_'+method+'.pickle','rb') as file:
                results[env][method] = pickle.load(file)
        else:
            results[env][method] = {}

            raw = []
            for n in range(NEXP):
                raw.append(pd.read_csv('cpu_mem_profile/'+method+'_'+env+'_'+str(n)+'.csv',sep=';'))
            
            formated = []
            for n in range(NEXP):
                raw[n]['time_s'] = raw[n]['time_s'].round(0)
                formated.append(raw[n].groupby('time_s').mean())
                formated[-1] = formated[-1].reset_index()

            lst = [formated[n]['cpu_percent'] for n in range(NEXP)]
            cpu_df = pd.concat(lst, axis=1, ignore_index=True)
            cpu_df.fillna(method='ffill', inplace=True)
            lst = [formated[n]['rss_mb'] for n in range(NEXP)]
            mem_df = pd.concat(lst, axis=1, ignore_index=True)
            mem_df.fillna(method='ffill', inplace=True)
            print(mem_df.iloc[-20:,:])

            m, l, h = mean_confidence_interval(cpu_df)
            results[env][method]['cpu_mean'] = m
            results[env][method]['cpu_low'] = l
            results[env][method]['cpu_high'] = h

            m, l, h = mean_confidence_interval(mem_df)
            results[env][method]['mem_mean'] = m
            results[env][method]['mem_low'] = l
            results[env][method]['mem_high'] = h

            with open('./cpu_mem_profile/pickles/'+env+'_'+method+'.pickle','wb') as file:
                pickle.dump(results[env][method],file)

















print('Plotting Single Results')
for env in ENVS:
    print('-',env)
    
    plot_lines(results[env],'cpu_mean','cpu_low','cpu_high','Time (s)','CPU (%)','./plots/cpu_mem_profile/cpu_single/'+env+'_CPU.pdf')
    plot_lines({method:results[env][method] for method in results[env] if method != 'rhopomcp'},'cpu_mean','cpu_low','cpu_high','Time (s)','CPU (%)','./plots/cpu_mem_profile/cpu_single/'+env+'_CPU_zoom.pdf')
    plot_box(results[env],'cpu_mean','CPU (%)','./plots/cpu_mem_profile/cpu_single/'+env+'_CPU_box.pdf')

    plot_lines(results[env],'mem_mean','mem_low','mem_high','Time (s)','RAM (MB)','./plots/cpu_mem_profile/mem_single/'+env+'_RAM.pdf')
    plot_lines({method:results[env][method] for method in results[env] if method != 'rhopomcp'},'mem_mean','mem_low','mem_high','Time (s)','RAM (MB)','./plots/cpu_mem_profile/mem_single/'+env+'_RAM_zoom.pdf')
    plot_box(results[env],'mem_mean','RAM (MB)','./plots/cpu_mem_profile/mem_single/'+env+'_RAM_box.pdf')

















print('Plotting RAM Summary')

fig, axs = plt.subplots(4,3,figsize=(20,20))
mean_label = 'mem_mean'
low_label = 'mem_low'
high_label = 'mem_high'
ylabel = 'RAM (MB)'
figname = './plots/cpu_mem_profile/RAM.pdf'
for i in range(len(ENVS)):
    env = ENVS[i]
    ### mean overal RAM
    maxx,minx = -np.inf, np.inf
    maxy,miny = -np.inf, np.inf
    for method in METHODS:
        x = [i for i in range(len(results[env][method][mean_label]))]
        y = results[env][method][mean_label]
        y_bci = results[env][method][low_label]
        y_hci = results[env][method][high_label]

        maxx,minx = max(maxy,x[-1]), min(miny,x[0])
        maxy,miny = max(maxy,max(y_hci)), min(miny,min(y_bci))

        axs[i,0].plot(x,y,label=METHODS_DICT[method],
            color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
        axs[i,0].fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
        axs[i,0].set_xticks([float('%.2f' %num)for num in np.linspace(minx,maxx,5)])
        axs[i,0].set_yticks([float('%.2f' %num) for num in np.linspace(miny,maxy,5)])
        axs[i,0].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        axs[i,0].set_ylabel(ENVS_DICT[env]+'\n'+ylabel,fontdict=FONT_DICT)
        if i == len(ENVS)-1:
            axs[i,0].set_xlabel('Time (s)',fontdict=FONT_DICT)
        if i == 0:
            axs[i,0].legend(fontsize=LEGEND_FONTSIZE)
    ### mean zoomed RAM
    maxx,minx = -np.inf,np.inf
    maxy,miny = -np.inf,np.inf
    for method in METHODS:
        if method != 'rhopomcp':
            x = [i for i in range(len(results[env][method][mean_label]))]
            y = results[env][method][mean_label]
            y_bci = results[env][method][low_label]
            y_hci = results[env][method][high_label]

            maxx,minx = max(maxx,x[-1]), min(minx,x[0])
            maxy,miny = max(maxy,max(y_hci)), min(miny,min(y_bci))

            axs[i,1].plot(x,y,label=METHODS_DICT[method],
                color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
            axs[i,1].fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
            axs[i,1].set_xticks([float('%.2f' %num)for num in np.linspace(minx,maxx,5)])
            axs[i,1].set_yticks([float('%.2f' %num) for num in np.linspace(miny,maxy,5)])
            axs[i,1].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            if i == len(ENVS)-1:
                axs[i,1].set_xlabel('Time (s)',fontdict=FONT_DICT)
    ### boxplot RAM
    data = []
    maxx,minx = -np.inf,np.inf
    maxy,miny = -np.inf,np.inf
    for method in METHODS:
        data.append(results[env][method][mean_label])
    boxprops = dict(linestyle='--', linewidth=1.5, color='black')
    flierprops = dict(marker='o', markerfacecolor='tab:cyan', markersize=5, alpha=0.5)
    medianprops = dict(linestyle='-.', linewidth=1.5, color='tab:blue')
    whiskerprops = dict(linestyle='-', linewidth=1.5, color='tab:green')
    boxplot = axs[i,2].boxplot(data, boxprops=boxprops, flierprops=flierprops,
                     medianprops=medianprops, whiskerprops=whiskerprops)
    
    xticks = METHODS_DICT.values()
    axs[i,2].set_xticklabels(xticks,rotation=10)
    axs[i,2].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    min_value = min(min(cap.get_ydata()) for cap in boxplot['caps'])
    max_value = max(max(cap.get_ydata()) for cap in boxplot['caps'])

    # Include outliers (fliers)
    if 'fliers' in boxplot:
        for flier in boxplot['fliers']:
            outliers = flier.get_ydata()
            if len(outliers) > 0:
                min_value = min(min_value, min(outliers))
                max_value = max(max_value, max(outliers))
    axs[i,2].set_yticks([float('%.2f' %num) for num in np.linspace(min_value,max_value,5)])
fig.tight_layout()
plt.savefig(figname)


















print('Plotting Scale')
FIG_COUNTER += 10
fig = plt.figure(FIG_COUNTER,figsize=(12,6))
mean_label = 'mem_mean'
low_label = 'mem_low'
high_label = 'mem_high'
ylabel = 'RAM (GB)'
figname = './plots/cpu_mem_profile/RAM_scale.pdf'

ry = {}
ry_bci = {}
ry_hci = {}
for method in METHODS:
    ry[method] = []
    ry_bci[method] = []
    ry_hci[method] = []
    for i in range(len(ENVS)):
        env = ENVS[i]
        maxx,minx = -np.inf, np.inf
        maxy,miny = -np.inf, np.inf

        y = results[env][method][mean_label]
        y_bci = results[env][method][low_label]
        y_hci = results[env][method][high_label]

        maxy,miny = max(maxy,max(y_hci)), min(miny,min(y_bci))

        ry[method].append(np.mean(y))
        ry_bci[method].append(np.mean(y_bci))
        ry_hci[method].append(np.mean(y_hci))

    x = [ENVS_DICT[env] for env in ENVS]
    plt.plot(x,
            ry[method],label=METHODS_DICT[method],
        color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=1,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
    plt.fill_between(x,ry_bci[method],ry_hci[method],color=COLORS[method],alpha=0.4)

plt.ylabel(ylabel,fontdict=FONT_DICT)
plt.xticks(fontsize=TICK_FONTSIZE,rotation=10)
plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
plt.legend(loc='best',fontsize=LEGEND_FONTSIZE)
fig.tight_layout()
plt.savefig(figname)













print('Plotting CPU Summary')
fig, axs = plt.subplots(4,3,figsize=(20,20))
mean_label = 'cpu_mean'
low_label = 'cpu_low'
high_label = 'cpu_high'
ylabel = r'CPU (%)'
figname = './plots/cpu_mem_profile/CPU.pdf'
for i in range(len(ENVS)):
    env = ENVS[i]
    ### mean overal CPU
    maxx,minx = -np.inf, np.inf
    maxy,miny = -np.inf, np.inf
    for method in METHODS:
        x = [i for i in range(1,len(results[env][method][mean_label]))]
        y = np.array(results[env][method][mean_label])[1:]
        y_bci = np.array(results[env][method][low_label])[1:]
        y_hci = np.array(results[env][method][high_label])[1:]

        maxx,minx = max(maxx,x[-1]), min(minx,x[0])
        maxy,miny = max(maxy,max(y_hci)), min(miny,min(y_bci))

        axs[i,0].plot(x,y,label=METHODS_DICT[method],
            color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
        axs[i,0].fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
        axs[i,0].set_xticks([float('%.2f' %num)for num in np.linspace(minx,maxx,5)])
        axs[i,0].set_yticks([float('%.2f' %num) for num in np.linspace(miny,maxy,5)])
        axs[i,0].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        axs[i,0].set_ylabel(ENVS_DICT[env]+'\n'+ylabel,fontdict=FONT_DICT)
        if i == len(ENVS)-1:
            axs[i,0].set_xlabel('Time (s)',fontdict=FONT_DICT)
        if i == 0:
            axs[i,0].legend(fontsize=LEGEND_FONTSIZE)
    ### mean zoomed CPU
    maxx,minx = -np.inf,np.inf
    maxy,miny = -np.inf,np.inf
    for method in METHODS:
        if method != 'rhopomcp':
            x = [i for i in range(1,len(results[env][method][mean_label]))]
            y = np.array(results[env][method][mean_label])[1:]
            y_bci = np.array(results[env][method][low_label])[1:]
            y_hci = np.array(results[env][method][high_label])[1:]

            maxx,minx = max(maxx,x[-1]), min(minx,x[0])
            maxy,miny = max(maxy,max(y_hci)), min(miny,min(y_bci))

            axs[i,1].plot(x,y,label=METHODS_DICT[method],
                color=COLORS[method],marker=MARKER[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,linewidth=LINEWIDTH,linestyle=LINESTYLE[method], markeredgecolor='black')
            axs[i,1].fill_between(x,y_bci,y_hci,color=COLORS[method],alpha=0.4)
            axs[i,1].set_xticks([float('%.2f' %num)for num in np.linspace(minx,maxx,5)])
            axs[i,1].set_yticks([float('%.2f' %num) for num in np.linspace(miny,maxy,5)])
            axs[i,1].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            if i == len(ENVS)-1:
                axs[i,1].set_xlabel('Time (s)',fontdict=FONT_DICT)
    ### boxplot CPU
    data = []
    maxx,minx = -np.inf,np.inf
    maxy,miny = -np.inf,np.inf
    for method in METHODS:
        data.append(np.array(results[env][method][mean_label])[1:])
    boxprops = dict(linestyle='--', linewidth=1.5, color='black')
    flierprops = dict(marker='o', markerfacecolor='tab:cyan', markersize=5, alpha=0.5)
    medianprops = dict(linestyle='-.', linewidth=1.5, color='tab:blue')
    whiskerprops = dict(linestyle='-', linewidth=1.5, color='tab:green')
    boxplot = axs[i,2].boxplot(data, boxprops=boxprops, flierprops=flierprops,
                     medianprops=medianprops, whiskerprops=whiskerprops)
    
    xticks = METHODS_DICT.values()
    axs[i,2].set_xticklabels(xticks,rotation=10)
    axs[i,2].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    min_value = min(min(cap.get_ydata()) for cap in boxplot['caps'])
    max_value = max(max(cap.get_ydata()) for cap in boxplot['caps'])

    # Include outliers (fliers)
    if 'fliers' in boxplot:
        for flier in boxplot['fliers']:
            outliers = flier.get_ydata()
            if len(outliers) > 0:
                min_value = min(min_value, min(outliers))
                max_value = max(max_value, max(outliers))
    axs[i,2].set_yticks([float('%.2f' %num) for num in np.linspace(min_value,max_value,5)])
fig.tight_layout()
plt.savefig(figname)