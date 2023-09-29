import mne
from autoreject import get_rejection_threshold, Ransac
import matplotlib.pyplot as plt

from constants import SCRATCH_FOLDER, RAW_FILENAME, RAW_PREP_FILENAME, ICA_FILENAME, BAD_CHANNELS_FILENAME
from configs import configs, exp_folder



def preprocess(subject):

    infile = SCRATCH_FOLDER / subject / RAW_FILENAME
    icafile = SCRATCH_FOLDER / subject / ICA_FILENAME
    badsfile = SCRATCH_FOLDER / subject / BAD_CHANNELS_FILENAME
    outfile = SCRATCH_FOLDER / subject / RAW_PREP_FILENAME

    if outfile.exists() and not configs['overwrite']:
        print(f'{outfile.name} already exists for {subject}. Skipping.')
        return None

    # First prep
    raw = mne.io.read_raw_fif(infile, preload=True).copy()
    filtered = filter(raw)

    if configs['run_ransac']:
        if badsfile.exists() and not configs['overwrite']:
            print("skipping ransac")
        else:
            bads = ransac_bad_channels(filtered)
            with open(badsfile, 'w') as f:
                f.write('\n'.join(bads))

    # Load bad channels 
    with open(badsfile, 'r') as f:
        bads = f.read().splitlines()
    filtered.info['bads'] = bads

    if configs['run_ICA']:
        if icafile.exists() and not configs['overwrite']:
            print("skipping ICA")
        else:
            ica = run_ICA(filtered)
            ica.save(icafile, overwrite=True)
    
    ica = mne.preprocessing.read_ica(icafile)
    filtered = ica.apply(filtered)
    
    filtered.save(outfile, overwrite=True)

def filter(raw):
    raw = raw.filter(l_freq=configs['l_freq'], h_freq=configs['h_freq'])
    return raw
    
def run_ICA(raw):
    
    # filter high frequencies to avoid drift
    raw = raw.filter(l_freq=1, h_freq=None)
    
    # Set parameters from configs
    ica_variances_explained = configs['ica_variances_explained']
    random_state = configs['random_state']
    t_step = configs['segmentation_tstep']
    
    # Get subject info
    s_info = raw.info['subject_info']['his_id']

    # Get rejection threshold
    events = mne.make_fixed_length_events(raw, duration=t_step)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=t_step, baseline=None, preload=True, reject=None, reject_by_annotation=True)
    # threshold = get_rejection_threshold(epochs)
    threshold = configs['reject_blink']
    print(f'Rejection threshold: {threshold}')

    # Drop bad epochs based on threshold
    epochs.drop_bad(reject=threshold)

    # Run ICA
    ica = mne.preprocessing.ICA(n_components=ica_variances_explained,
                                method=configs['ica_method'],
                                random_state=random_state) 
    ica.fit(epochs, tstep=t_step)

    # Plot components and sources
    # Set title for plots
    title = f'ICA - {s_info}'
    components_fig = ica.plot_components(show=False, title=f'{title} - components')[0]
    components_fig.savefig(exp_folder / 'plots' / f'{title} - components.png')

    sources_fig = ica.plot_sources(raw, show=True, title=f'{title} - sources', show_scrollbars=True, start=200, stop=250)
    sources_fig.savefig(exp_folder / 'plots' / f'{title} - sources.png')

    # Manually select components to exclude
    blink_component = input()
    ica.exclude = [int(x) for x in blink_component.split(',')]

    return ica

def ransac_bad_channels(raw):
    tstep = configs['bad_tstep']
    events = mne.make_fixed_length_events(raw, duration = tstep)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=tstep, baseline=None, preload=True)

    ransac = Ransac(n_jobs=-1)
    ransac = ransac.fit(epochs)

    print('Bad channels detected: ')
    print('\n'.join(ransac.bad_chs_)) # list of bad channels
    
    del epochs
    return ransac.bad_chs_