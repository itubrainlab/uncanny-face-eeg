import mne
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from analysis import read_time_table, find_peak, make_evo_dict, ANOVA_test

from constants import SCRATCH_FOLDER, ERP_FILENAME, SUBJECTS, EPOCHS_FILENAME, CONDITIONS, BEHAVE_CONDITIONS
from configs import configs, exp_folder

def ss_plot(subject):

    if configs['evokeds']:
        # plot_evoked_imgs(subject)
        plot_evoked(subject)
    if configs['compare_evokeds']:
        plot_differences(subject)
    if configs['compare_topos']:
        plot_topos(subject)
    if configs['plot_behaviour']:
        plot_behaviour(subject)

def group_plot():

    csv_file = SCRATCH_FOLDER / ERP_FILENAME

    if configs['group_erp']:
        erp_plot()

    if configs['violins']:
        plot_violins(df)

    if configs['statistics']:
        if not configs['overwrite'] and csv_file.exists():
            print('Loading ERP data')
            df = pd.read_csv(csv_file)
        else:
            df = make_big_df(csv_file)
        print('Running ANOVA at for all subjects')
        for component in configs['components']:
            print(f'{component}\n')
            df_comp = df[df['component'] == component]
            ANOVA_test(df_comp)

def make_big_df(csv_file):
    dfs = []
    for subject in SUBJECTS:
        for component in configs['components']:
            infile = SCRATCH_FOLDER / subject / f'{component}_ERP.csv'
            df = pd.read_csv(infile)
            df['subject'] = subject
            df['component'] = component
            dfs.append(df)

    df = pd.concat(dfs)

    # save data    
    df.to_csv(csv_file, index=True)
    return df

def plot_violins(df):

    conditions = df.category.unique()
    x = range(len(conditions))

    for subject in SUBJECTS:
        
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 16))
        fig.suptitle(f'{subject} ERP measures', fontsize=20)

        df_sub = df[df['subject'] == subject]

        for i, component in enumerate(configs['components']):
            axes[i,0].set_ylabel(component)

            df_comp = df_sub[df_sub['component'] == component]

            for j, measure in enumerate(['peakamp', 'average', 'latency']):
                axes[0,j].set_title(measure)
                
                ax = axes[i,j]
                ax.set_xticks(x, conditions, rotation='horizontal')
                data_to_plot = []
                for cond in conditions:
                    data = df_comp[df_comp['category'] == cond][measure].values
                    data_to_plot.append(data)
                ax.violinplot(data_to_plot, positions=x, showmeans=True, showextrema=False)
        plt.tight_layout()
        fig.savefig(exp_folder / 'plots' / f'violins_{subject}.png')

def plot_evoked(subject):

    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    epochs = mne.read_epochs(infile).copy()
    if configs['standard_ref'] != None:
        epochs = epochs.set_eeg_reference(configs['standard_ref'])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(f'{subject} Evoked signals', fontsize=20)

    for i, component in enumerate(configs['components']):
        ax = axes[int(i<2), i%2]
        ax.set_title(component)
        if configs['custom_times']:
            tmin, tmax = read_time_table(subject, component, unit='s')
        else:
            tmin, tmax = configs[component]['times']
        epochs_comp = epochs.copy().pick_channels(configs[component]['ch_picks']).crop(tmin-0.1, tmax+0.1)
        evokeds = make_evo_dict(epochs_comp, include_combined=False, diff=configs['on_diff'])
        mne.viz.plot_compare_evokeds(evokeds, show=False, axes=ax, show_sensors=False, title=component, combine='mean')
        ax.axvspan(xmin=tmin, xmax=tmax, facecolor='gray', alpha=0.2)
    plt.tight_layout()
    fig.savefig(exp_folder / 'plots' / f'{subject}_compare_evoked.png') 
    plt.close()

def plot_evoked_imgs(subject):
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    epochs = mne.read_epochs(infile)
    if configs['standard_ref'] != None:
        epochs.set_eeg_reference(configs['standard_ref'])

    conditions = epochs.metadata.category.unique()
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(24, 16), sharey='row')
    fig.suptitle(f'{subject} ERP signals', fontsize=20)

    for i, component in enumerate(configs['components']):
        axes[i,0].set_ylabel(component)
        if configs['custom_times']:
            tmin, tmax = read_time_table(subject, component)
        else:
            tmin, tmax = configs[component]['times']

        epochs_comp = epochs.copy().pick_channels(configs[component]['ch_picks'])

        for j, condition in enumerate(conditions):
            axes[0,j].set_title(condition)
            ax = axes[i,j]
            epochs_cond = epochs_comp[f'category == "{condition}"']
            make_butterfly(epochs_cond, ax, tmin, tmax, configs[component]['mode'])

    plt.tight_layout()
    fig.savefig(exp_folder / 'plots' / f'{subject}_img_evoked.png')
    plt.close()

def make_butterfly(epochs, ax, tmin, tmax, mode):
    colors = cm.get_cmap('tab20').colors
    imgs = epochs.metadata.image_id.unique()

    for i, img in enumerate(imgs):
        color = colors[0]
        data = epochs[epochs.metadata.image_id == img].average().get_data()[0,:] * 1e6
        ax.plot(epochs.times, data, alpha=1, color=color)
        peak, lat, avg = find_peak(data, epochs.times, tmin, tmax, mode)
        
        ax.plot(lat, peak, marker='*', color=color)

    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(0, color='k', linestyle='-')
    ax.axvspan(xmin=tmin, xmax=tmax, facecolor='gray', alpha=0.2)
    
    evoked = epochs.average().get_data()[0,:] * 1e6
    ax.plot(epochs.times, evoked, alpha=1, color='k', linewidth=3)
    offset = 0.1
    ax.set_xlim(tmin-offset, tmax+offset)

def plot_differences(subject):
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    epochs = mne.read_epochs(infile).copy()
    if configs['standard_ref'] != None:
        epochs.set_eeg_reference(configs['standard_ref'])

    if configs['on_diff']:
        evokeds = make_evo_dict(epochs, diff=True)
        colordic = {bcond: i for i, bcond in enumerate(evokeds.keys())}
        fig = mne.viz.plot_compare_evokeds(evokeds, picks='eeg', colors=colordic, axes='topo', show=False)
    elif configs['earlylate']:
        evokeds = make_evo_dict(epochs, include_combined=False, early=True)
        fig = mne.viz.plot_compare_evokeds(evokeds, picks='eeg', colors=dict(early=1, late=0), axes='topo', show=False)
    elif configs['on_behave']:
        evokeds = make_evo_dict(epochs, include_combined=False, behave=True)
        colordic = {bcond: i for i, bcond in enumerate(BEHAVE_CONDITIONS)}
        fig = mne.viz.plot_compare_evokeds(evokeds, picks='eeg', colors=colordic, axes='topo', show=False)
    else:
        evokeds = make_evo_dict(epochs, include_combined=True)
        fig = mne.viz.plot_compare_evokeds(evokeds, picks='eeg', axes='topo', show=False)

    filename = exp_folder /  'plots' / f'{subject}_all_topo.png'
    fig[0].savefig(filename)
    # return None

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16)) 
    def custom_func(x):
        return x.min(axis=1)
    for i, combine in enumerate(['mean', 'median', 'gfp', custom_func]):
        ax = axes[int(i<2), i%2]
        lpp_picks = configs['N400']['ch_picks']
        mne.viz.plot_compare_evokeds(evokeds, picks=lpp_picks, combine=combine, axes=ax, show=False)[0]
    title = f'{subject}_compare'
    fig.suptitle = title
    fig.savefig(exp_folder / 'plots' / (title+'.png'))

def plot_topos(subject):
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    epochs = mne.read_epochs(infile).copy()
    if configs['standard_ref'] != None:
        epochs.set_eeg_reference(configs['standard_ref'])
    if configs['on_diff']:
        evokeds = make_evo_dict(epochs, diff=True)
    elif configs['earlylate']:
        evokeds = make_evo_dict(epochs, early=True)
    elif configs['on_behave']:
        evokeds = make_evo_dict(epochs, behave=True)
    else:
        evokeds = make_evo_dict(epochs, include_combined=True)
    
    num_evos = len(evokeds)
    times = [0.75, 0.2, 0.3, 0.4, 0.5, 0.6]
    avges = [0.025, 0.05, 0.1, 0.1, 0.1, 0.1]
    fig, axes = plt.subplots(nrows=num_evos, ncols=len(times), figsize=(4*len(times), 4*num_evos)) 
    title = f'{subject}_topos'
    
    for i, cond in enumerate(evokeds.keys()):
        ax = axes[i,:]
        evokeds[cond].plot_topomap(
            ch_type='eeg', times=times, average=avges, colorbar=False, axes=ax, show=False
        )
        ax[0].set_ylabel(cond)
    fig.suptitle = title
    fig.savefig(exp_folder / 'plots' / (title+'.png'))

def plot_behaviour(subject):
    import numpy as np
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    epochs = mne.read_epochs(infile).copy()

    exp_time = 0 if configs['include_short'] else 0.2
    epos = epochs[f'exposure_time >= {exp_time}']
    print(f'epos used for short: {len(epos)}')
    data = epos.metadata
    dfs = {cond : data[data.category == cond] for cond in CONDITIONS}
    ratios = {cond : [] for cond in CONDITIONS}
    for cond, df in dfs.items():
        for img in df.image_id.unique():
            img_df = df[df.image_id == img]
            ratio1 = len(img_df[img_df.keypress == "1"]) / len(img_df)
            ratio2 = len(img_df[img_df.keypress == "2"]) / len(img_df)
            ratio3 = len(img_df[img_df.keypress == "3"]) / len(img_df)
            ratio_tuple = (ratio1, ratio2, ratio3)
            ratios[cond].append(ratio_tuple)
    width = 0.8
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 12), sharey=True)

    for i, cond in enumerate(CONDITIONS):
        ratio = ratios[cond]
        ratio.sort(reverse=True, key=lambda a: a[0])
        x = {}
        for j, behav_cond in enumerate(BEHAVE_CONDITIONS):
            x[behav_cond] = [r[j] for r in ratio]
        N = len(ratio)
        ind = np.arange(N)
        
        ax = axes[i]
        
        ax.bar(ind, x[BEHAVE_CONDITIONS[0]], width, color='y')
        ax.bar(ind, x[BEHAVE_CONDITIONS[1]], width, bottom=x[BEHAVE_CONDITIONS[0]], color='b')
        ax.bar(ind, x[BEHAVE_CONDITIONS[2]], width, bottom=x[BEHAVE_CONDITIONS[1]], color='r')
        
        ax.set_title(cond)
        ax.set_xticks([])
    axes[-1].legend(labels=BEHAVE_CONDITIONS)
    axes[0].set_ylabel('Ratio')
    fig.tight_layout()
    title = f"{subject}_behavior_long.png"
    fig.savefig(exp_folder / 'plots' / title)
    #plt.show()
    
def erp_plot():

    if configs['on_behave']:
        conds = BEHAVE_CONDITIONS
    else: 
        conds = CONDITIONS

    all_evos = {}

    for subject in SUBJECTS:
        infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
        epochs = mne.read_epochs(infile).copy()
        epochs = epochs.interpolate_bads()
        if configs['standard_ref'] != None:
            epochs = epochs.set_eeg_reference(ref_channels=configs['standard_ref'])
        evokeds = make_evo_dict(epochs, include_combined=False, behave=configs['on_behave'], diff=configs['on_diff'])
        conds = evokeds.keys()
        for cond in conds:
            temp = all_evos.get(cond,[])
            temp.append(evokeds[cond])
            all_evos[cond] = temp

    colordic = {ccond: i for i, ccond in enumerate(conds)}
    for comp in configs['components']:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        title = f'{comp} comparison'
        mne.viz.plot_compare_evokeds(all_evos,
                                    combine='mean',
                                    legend='upper left',
                                    picks=configs[comp]['ch_picks'],
                                    show_sensors='lower right',
                                    axes=ax,
                                    colors=colordic,
                                    title=title,
                                    show=False
                                    )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_ylim([-8, 8])
        ax.set_yticks(range(-7, 8, 2))  # set y-axis ticks at -9, -6, -3, 0, 3, 6, and 9
        ax.yaxis.grid(True, linestyle='--', color='lightgray')  # extend gridlines to -7 and 7
        fig.savefig(exp_folder / 'plots' / (title+'.png'))