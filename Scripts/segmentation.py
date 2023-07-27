import mne
import pandas as pd
from autoreject import AutoReject
import matplotlib.pyplot as plt

from constants import SCRATCH_FOLDER, RAW_FOLDER, RAW_PREP_FILENAME, EPOCHS_FILENAME, CATEGORY_MAPPING, N_SESSIONS
from configs import configs, exp_folder

def segment(subject):

    infile = SCRATCH_FOLDER / subject / RAW_PREP_FILENAME
    outfile = SCRATCH_FOLDER / subject / EPOCHS_FILENAME

    if outfile.exists() and not configs['overwrite']:
        print(f'{outfile} already exists. Skipping.')
        return None

    # Load raw data
    raw = mne.io.read_raw_fif(infile).copy()

    # Set events
    events = mne.find_events(raw, stim_channel='trigger', verbose=False)

    # Epoch
    epochs = mne.Epochs(raw,
                        events,
                        tmin=configs['e_tmin'],
                        tmax=configs['e_tmax'],
                        baseline=None,
                        preload=True)

    if configs['tshift'] != 0:
        epochs.shift_time(tshift=configs['tshift'])
    
    # Baseline correct
    epochs.apply_baseline(baseline=configs['e_baseline'])

    # Locate behaviour files
    metas = []
    for session in range(1,N_SESSIONS+1):
        behaviour_file = RAW_FOLDER / subject / str(session) / 'behave.csv'
    
        # Read behave file and add subjcet and session info
        metadata = pd.read_csv(behaviour_file)
        metadata['subject'] = subject
        metadata['session'] = session

        # Add category info to metadata
        metadata['category'] = pd.Series([CATEGORY_MAPPING(img_id) for img_id in metadata['image_id']])
        metas.append(metadata)
    metadata = pd.concat(metas)

    metadata['keypress'] = metadata.keypress.astype(str)

    # Add metadata to epochs
    epochs.metadata = metadata

    # Auto reject epochs
    ar = AutoReject(random_state=configs['random_state'], n_jobs=-1)
    epochs, log = ar.fit_transform(epochs, return_log=True)
    fig = log.plot('horizontal', show=False)
    fig.savefig(exp_folder / 'plots' / f'{subject}_rejectlog.png')

    # Save epochs
    epochs.save(outfile, overwrite=True)