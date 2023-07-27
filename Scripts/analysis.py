import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from constants import SCRATCH_FOLDER, EPOCHS_FILENAME, CATEGORY_MAPPING, TIME_TABLE_FILENAME, CONDITIONS, BEHAVE_CONDITIONS
from configs import configs, exp_folder

def analyse(subject):

    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME
    # epochs = mne.read_epochs(infile)
    
    # Run GFP analysis
    if configs['GFP']:
        GFP_analysis(epochs)

    # Run ERP analysis
    if configs['ERP']:
        for component in configs['components']:
            ERP_analysis(epochs, component)

    # Compare evoked responses between conditions
    if configs['EVK']:
        evokeds = make_evo_dict(epochs)
        for conds in evokeds.keys():
            title = f'{subject} {conds}.png'
            fig = evokeds[conds].plot(show=True)
            fig.savefig(exp_folder / 'plots' / title)

    if configs['DISS']:
        global_diss(epochs, subject)
    
    if configs['N400_test']:
        N400_test(subject)
        # behave_test(subject)
    if configs['N170_test']:
        N170_test(subject)
    #for component in configs['components']:
    #    window_check(epochs, component)

    if configs['DECO']:
        temp_decoding(epochs, subject)

    if configs['GFP_TEST']:
        GFP_test(epochs, subject)

def GFP_test(epochs, subject):

    evokeds = make_evo_dict(epochs, include_combined=False, on_images=True)

    times = epochs.times
    time_index = epochs.time_as_index(times)
    p_vals = []
    for t in time_index:
        categories = []
        GFPs = []
        for img_id, evo in evokeds.items():
            data = evo.get_data()[:,t]
            GFP = data.std(axis=0, ddof=0)

            category = CATEGORY_MAPPING(img_id, output='str')
            GFPs.append(GFP)
            categories.append(category)
        df = pd.DataFrame(list(zip(GFPs, categories)),
                          columns =['GFP', 'cat'])

        model = ols('GFP ~ C(cat)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p = anova_table['PR(>F)'][0]
        p_vals.append(1-p)
   
    fig, ax = plt.subplots()
    plt.plot(times, p_vals, color='blue', alpha=0.4)
    plt.fill_between(times, p_vals, 0, color='blue', alpha=0.4)
    plt.xlabel('Time (S)')
    plt.ylim(0.90, 1)
    plt.ylabel('p value')
    plt.margins(x=0)
    fig.savefig(exp_folder / 'plots' / f'{subject}_GFP_test.png') 

def GFP_normalise(evoked):
    # Returns data, not evoked
    picks = mne.pick_types(evoked.info, eeg=True, stim=False, exclude='bads')
    data = evoked.get_data(picks=picks)
    GFP = data.std(axis=0, ddof=0)
    return data / GFP

def global_diss(epochs, subject):
    from random import shuffle
    
    epos = epochs.copy().crop(tmin=-0.1)[f'category != ""']

    evokeds = make_evo_dict(epos)

    real_diss = calc_diss(evokeds)
    
    epochs_categories = epos.metadata.category.values

    simulated_diss = []
    for _ in range(configs['DISS_permutations']):
        shuffle(epochs_categories)
        epos.metadata["shuffled_category"] = epochs_categories
        evokeds = make_evo_dict(epos)
        diss = calc_diss(evokeds)
        simulated_diss.append(diss)
    #print(simulated_diss, real_diss)
    pvalue_graph(real_diss, simulated_diss, epos.times, subject)

    # fig, axs = plt.subplots(nrows=2, ncols=1)
    # ax = axs[0]
    # ax.plot(epos.times, corr, label='CORR')
    # ax.plot(epos.times, human_semi, label='DISS')
    # ax.legend()
    # ax.margins(x=0)

    # fig.suptitle(f'{subject}: dissimilarity analysis of human/semi')
    # plt.show()

def pvalue_graph(diss_actual, diss_empirical, times, subject):
    p_value = []
    z_score = []
    
    for t in range(len(diss_actual)):
        values = [diss_empirical[i][t] for i in range(configs['DISS_permutations'])]
        mean = np.mean(values)
        std = np.std(values)
        
        # Compute the Z-score
        score = (diss_actual[t] - mean)/std
        z_score.append(score)
        
        # Compute the p-value
        p = stats.norm.sf(abs(score))
        p_value.append(p)
        
    # Transfor lists into np.aray objects
    z_score = np.array(z_score)
    p_value = 1 - np.array(p_value)
    
    # Plotiing p-values for each time instant
    fig, ax = plt.subplots()
    plt.plot(times, p_value, color='blue', alpha=0.4)
    plt.fill_between(times, p_value, 0, color='blue', alpha=0.4)
    plt.xlabel('Time (S)')
    plt.ylim(0.90, 1)
    plt.ylabel('p value')
    plt.margins(x=0)
    fig.savefig(exp_folder / 'plots' / f'diss_matrix_{subject}.png') 

def calc_diss(evokeds):
    import numpy as np
    normed_evokeds = {c : GFP_normalise(evo) for c, evo in evokeds.items()}
    human_semi = normed_evokeds['human'] - normed_evokeds['semi-realistic']
    human_semi = human_semi**2
    human_semi = human_semi.mean(axis=0)
    diss = np.sqrt(human_semi)
    return diss
    corr = (2-(diss**2))/2

def GFP_analysis(epochs):

    s_id = epochs.metadata['subject'].unique()[0]

    evokeds = make_evo_dict(epochs)
    colors = {'animated': '#E9002D', 'human': '#FFAA00', 'semi-realistic': '#00B000'}

    # calc and plot GFP
    fig, axes = plt.subplots(2,1, figsize=(10,5), sharey=False, sharex=True)
    ax = axes[0]
    dax = axes[1]
    for cond, evoked in evokeds.items():
        picks = mne.pick_types(evoked.info, eeg=True, stim=False, exclude='bads')
        data = evoked.get_data(picks=picks)
        GFP = data.std(axis=0, ddof=0)
        ax.plot(evoked.times, GFP, color=colors[cond], label=cond)
    
        # Plot difference between conditions
        if cond == 'human':
            continue
        else:
            diff = GFP - evokeds['human'].get_data(picks=picks).std(axis=0, ddof=0)
            dax.plot(evoked.times, diff, color=colors[cond], label=cond)

    plot_times = (-0.05, 0.45)
    ax.legend()

    ax.set(xlabel='Time (S)', ylabel='GFP (µV)')
    dax.set(xlabel='Time (S)')
    
    ax.set_xlim(plot_times)  # in seconds
    ax.margins(x=0)
    
    # Set vertical line at 0
    ax.axvline(0, color='gray', linestyle='--')
    dax.axvline(0, color='gray', linestyle='--')

    # Set horizontal line at 0
    ax.axhline(0, color='k', linestyle='-')
    dax.axhline(0, color='k', linestyle='-')

    fig.suptitle(f'{s_id} GFP')
    fig.savefig(exp_folder / 'plots' / f'{s_id}_GFP.png')

def make_evo_dict(epochs, include_combined=False, on_images=False, behave=False, early=False, diff=False):
    evokeds = {}
    exp_time = 0 if configs['include_short'] else 0.2
    if include_combined:
        evokeds['all'] = epochs[f'exposure_time >= {exp_time}'].average()
    if diff:
        for cond in CONDITIONS:
            evokeds[cond] = epochs[f'category == "{cond}" and exposure_time >= {exp_time}'].average()
            weights = [1,-1]
            
        evos = {f'{CONDITIONS[0]} - {CONDITIONS[1]}' :  mne.combine_evoked([evokeds[CONDITIONS[0]], evokeds[CONDITIONS[1]]], weights),
                f'{CONDITIONS[0]} - {CONDITIONS[2]}' :  mne.combine_evoked([evokeds[CONDITIONS[0]], evokeds[CONDITIONS[2]]], weights),
                f'{CONDITIONS[1]} - {CONDITIONS[2]}' :  mne.combine_evoked([evokeds[CONDITIONS[1]], evokeds[CONDITIONS[2]]], weights)}
        return evos
    elif on_images:
        image_ids = np.unique(epochs.metadata.image_id.values)
        for img in image_ids:
            evokeds[img] = epochs[f'image_id == {img} and exposure_time >= {exp_time}'].average()
        return evokeds
    elif behave:
        for i, cond in enumerate(BEHAVE_CONDITIONS):
            epo = epochs[f'keypress == "{i+1}" and exposure_time >= {exp_time}']
            print(f'epos used on {cond}: {len(epo)}')
            evokeds[cond] = epo.average()
        return evokeds
    elif early:
        data = epochs.metadata
        img_reps = {}
        reps = []
        for index, trial in data.iterrows():
            img = trial['image_id']
            rep = img_reps.get(img, 0) +1
            img_reps[img] = rep
            reps.append(rep)
        data['repetitions'] = reps
        epochs.metadata = data
        evokeds['early'] = epochs['repetitions <= 3 and exposure_time >= {exp_time}'].average()#epochs[f'keypress == "1" and repetitions <= 3'].average()
        evokeds['late'] = epochs['repetitions >= 5 and exposure_time >= {exp_time}'].average()#epochs['keypress == "1" and repetitions >= 5'].average()
        return evokeds
    else:
        for cond in CONDITIONS:
            evokeds[cond] = epochs[f'category == "{cond}" and exposure_time >= {exp_time}'].average()    
    
    return evokeds

def ERP_analysis(epochs, component):

    print(f'\nRunning analysis on {component}\n')

    analysis_configs = configs[component]

    s_id = epochs.metadata['subject'].unique()[0]
    ERP_epochs = epochs.copy()
    if analysis_configs['ref'] != None:
        ERP_epochs = ERP_epochs.set_eeg_reference(ref_channels=analysis_configs['ref'])
    ERP_epochs = ERP_epochs.pick_channels(analysis_configs['ch_picks'])

    if configs['custom_times']:
        tmin, tmax = read_time_table(s_id, component)
    else:
        tmin, tmax = analysis_configs['times']

    # Find ERP peak for each image
    peaks = []
    latencies = []
    averages = []
    categories = []

    if configs['img_avg']:
        imgs = epochs.metadata.image_id.unique()
        for img in imgs:
            data = ERP_epochs[ERP_epochs.metadata.image_id == img].average().get_data()[0,:] * 1e6
            peak, lat, avg = find_peak(data, ERP_epochs.times, tmin, tmax, analysis_configs['mode'])
            
            category = CATEGORY_MAPPING(img, output='str')
            #if category != 'animated':
            categories.append(category)
            peaks.append(peak)
            averages.append(avg)
            latencies.append(lat)
    else:
        for i, e in enumerate(ERP_epochs.iter_evoked(copy=False)):
            data = e.get_data()[0,:] * 1e6
            peak, lat, avg = find_peak(data, e.times, tmin, tmax, analysis_configs['mode'])
            peaks.append(peak)
            averages.append(avg)
            latencies.append(lat)
            categories.append(ERP_epochs[i].metadata.category.values[0])

    df = pd.DataFrame({'average': averages,
                       'peakamp': peaks, 
                       'latency': latencies,
                       'category': categories})
    
    df.to_csv(SCRATCH_FOLDER / s_id / f'{component}_ERP.csv', index=False)
    
    if configs['statistics']:
        # Run stats
        ANOVA_test(df)

def plot_peak(evoked, lat, amp, analysis_configs):

    title = f'Peak: {round(lat)} {round(amp,1)}'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    evoked.plot(show=False, axes=ax, time_unit='ms', )
    ax.plot(lat, amp, marker='*', color='C6')
    ax.axvspan(xmin=analysis_configs['tmin'], xmax=analysis_configs['tmax'], facecolor='C0', alpha=0.3)
    #ax.axvspan(xmin=lat-10, xmax=lat+10, facecolor='C1', alpha=0.3)
    ax.set_xlim(analysis_configs['tmin']-100, analysis_configs['tmax']+100)  # Show zoomed in around peak

    fig.savefig(exp_folder / 'plots' / (title+'.png'))

def ANOVA_test(df):
    print('Average Test')
    model = ols('average ~ C(category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('Peak Amp Test')
    model = ols('peakamp ~ C(category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('Latency Test')
    model = ols('latency ~ C(category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

def Tukey(df):
    from bioinfokit.analys import stat
    # perform multiple pairwise comparison (Tukey's HSD)
    # unequal sample size data, tukey_hsd uses Tukey-Kramer test
    res = stat()
    res.tukey_hsd(df=df, res_var='average', xfac_var='category', anova_model='average ~ C(category)')
    print(res.tukey_summary)
    res = stat()
    res.tukey_hsd(df=df, res_var='peakamp', xfac_var='category', anova_model='peakamp ~ C(category)')
    print(res.tukey_summary)

def read_time_table(subject, component, unit='s'):
    # Import table of times
    table = pd.read_csv(SCRATCH_FOLDER / TIME_TABLE_FILENAME, sep='\t')
    subtable = table[table['Subject'] == subject]
    tmin, tmax = subtable[component].values[0].split('-')
    tmin = float(tmin)
    tmax = float(tmax)
    if unit == 's':
        tmin = tmin/1e3
        tmax = tmax/1e3
    return tmin, tmax

def find_peak(data, times, tmin, tmax, mode):
    data = data[(times > tmin) & (times < tmax)]
    times = times[(times > tmin) & (times < tmax)]
    if mode == 'neg':
        peak = data.min()
        lat = times[data.argmin()]
    elif mode == 'pos':
        peak = data.max()
        lat = times[data.argmax()]
    avg = data.mean()
    return peak, lat, avg

def temp_decoding(epochs, subject):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel, get_coef

    epochs = epochs.copy().crop(tmin=0.150,tmax=0.300)['category != "semi-realistic"']
    X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
    #epochs[f'category == "{cond}" and exposure_time > {exp_time}']
    y = epochs.metadata.category.values
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear'))
    )
    time_decod = SlidingEstimator(
        clf, n_jobs=None, scoring='roc_auc', verbose=True)
    time_decod.fit(X, y)

    coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    joint_kwargs = dict(ts_args=dict(time_unit='s'),
                        topomap_args=dict(time_unit='s'))
    fig = evoked_time_gen.plot_joint(times=np.arange(.150, .310, .030), title='patterns',
                                     show=False, **joint_kwargs)
    title = f'{subject}_decoding_patterns.png'
    fig.savefig(exp_folder / 'plots' / title)

    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X, y, cv=15, n_jobs=-1)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)


    # Plot
    fig, ax = plt.subplots()
    ax.plot(epochs.times, scores, label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Sensor space decoding')
    title = f'{subject}_decoding_AUC.png'
    fig.savefig(exp_folder / 'plots' / title)

def N400_test(subject):

    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME

    epochs = mne.read_epochs(infile).copy()

    if configs['custom_times']:
        tmin, tmax = read_time_table(subject, 'N400', unit='s')
    else:
        tmin, tmax = configs['N400']['times']

    epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax=True)

    exp_time = 0 if configs['include_short'] else 0.2

    cond_array = {}
    for cond in CONDITIONS:
        cond_epo = epochs[f'category == "{cond}" and exposure_time >= {exp_time}']
        
        # image_ids = np.unique(cond_epo.metadata.image_id.values)
        # min_dat = np.zeros(len(image_ids))
        # avg_dat = np.zeros(len(image_ids))
        # med_dat = np.zeros(len(image_ids))
        # qan_dat = np.zeros(len(image_ids))

        # for i, img in enumerate(image_ids):
        #     evo = cond_epo[f'image_id == {img}'].average()
        #     data = evo.get_data(picks=configs['N400']['ch_picks'])*1e6
        #     min_dat[i] = data.min()
        #     avg_dat[i] = data.mean()
        #     med_dat[i] = np.median(data)
        #     qan_dat[i] = np.quantile(data, 0.2)

        data = cond_epo.get_data(picks=configs['N400']['ch_picks'])*1e6

        min_dat = data.min(axis=(1,2))
        avg_dat = data.mean(axis=(1,2))
        med_dat = np.median(data, axis=(1,2))
        qan_dat = np.quantile(data, 0.2, axis=(1,2))

        avg_data = data.mean(axis=(1))
        med_avg = np.median(avg_data,axis=1)

        dat_array = {'avg' : avg_dat,
                     'min' : min_dat,
                     'med' : med_dat,
                     'qam' : qan_dat,
                     'avg_med' : med_avg
                     }
        
        cond_array[cond] = dat_array
    
    for measure in dat_array.keys():
        print('\n',measure)
        a = cond_array['human'][measure]
        b = cond_array['semi-realistic'][measure]
        res = stats.ttest_ind(a, b, alternative='greater')
        print(a.mean(),b.mean())
        print(res)
    
    # plt.plot(cond_array['human']['avg'])
    # plt.plot(cond_array['semi-realistic']['avg'])
    # plt.show()

def N170_test(subject):
    
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME

    epochs = mne.read_epochs(infile).copy()
    
    if configs['custom_times']:
        tmin, tmax = read_time_table(subject, 'N170', unit='s')
    else:
        tmin, tmax = configs['N170']['times']

    epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax=True)
    epochs = epochs.set_eeg_reference(ref_channels=configs['N170']['ref'])

    exp_time = 0 if configs['include_short'] else 0.2

    cond_array = {}
    for cond in CONDITIONS:
        cond_epo = epochs[f'category == "{cond}" and exposure_time >= {exp_time}']
        
        # image_ids = np.unique(cond_epo.metadata.image_id.values)
        # min_dat = np.zeros(len(image_ids))
        # avg_dat = np.zeros(len(image_ids))
        # med_dat = np.zeros(len(image_ids))
        # qan_dat = np.zeros(len(image_ids))

        # for i, img in enumerate(image_ids):
        #     evo = cond_epo[f'image_id == {img}'].average()
        #     data = evo.get_data(picks=configs['N400']['ch_picks'])*1e6
        #     min_dat[i] = data.min()
        #     avg_dat[i] = data.mean()
        #     med_dat[i] = np.median(data)
        #     qan_dat[i] = np.quantile(data, 0.2)

        data = cond_epo.get_data(picks=configs['N170']['ch_picks'])*1e6

        min_dat = data.min(axis=(1,2))
        avg_dat = data.mean(axis=(1,2))
        med_dat = np.median(data, axis=(1,2))
        qan_dat = np.quantile(data, 0.2, axis=(1,2))

        avg_data = data.mean(axis=(1))
        med_avg = np.median(avg_data,axis=1)

        dat_array = {'avg' : avg_dat,
                     'min' : min_dat,
                     'med' : med_dat,
                     'qam' : qan_dat,
                     'avg_med' : med_avg
                     }
        
        cond_array[cond] = dat_array
    
    for measure in dat_array.keys():
        a = cond_array['human'][measure]
        b = cond_array['animated'][measure]
        res = stats.ttest_ind(a, b, alternative='greater')
        print(measure, res)
        print(a.mean(),b.mean())

def behave_test(subject):
    
    infile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME

    epochs = mne.read_epochs(infile).copy()

    if configs['custom_times']:
        tmin, tmax = read_time_table(subject, 'N400', unit='s')
    else:
        tmin, tmax = configs['N400']['times']

    epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax=True)

    cond_array = {}
    evokeds = make_evo_dict(epochs, behave=True)

    for cond, evo in evokeds.items():

        data = evo.get_data(picks=configs['N400']['ch_picks'])*1e6

        min_dat = data.min(axis=(1,2))
        avg_dat = data.mean(axis=(1,2))
        med_dat = np.median(data, axis=(1,2))
        qan_dat = np.quantile(data, 0.2, axis=(1,2))

        avg_data = data.mean(axis=(1))
        med_avg = np.median(avg_data,axis=1)

        dat_array = {'avg' : avg_dat,
                     'min' : min_dat,
                     'med' : med_dat,
                     'qam' : qan_dat,
                     'avg_med' : med_avg
                     }
        
        cond_array[cond] = dat_array
    
    for measure in dat_array.keys():
        print('\n',measure)
        a = cond_array['alive'][measure]
        b = cond_array['normal'][measure]
        res = stats.ttest_ind(a, b, alternative='greater')
        print(a.mean(),b.mean())
        print(res)