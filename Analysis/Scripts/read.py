import os
import mne
import mat73 # Needed for reading newer .mat files
import numpy as np
import pandas as pd

from constants import (RAW_FOLDER, SCRATCH_FOLDER, CH_POSITIONS, RAW_FILENAME, N_SESSIONS, RAW_MAT_FILENAME, 
                       SFREQ, N_TRIALS_PER_BREAK)
from configs import configs, exp_folder


def make_scratch(id):
    # Making a scratch folder for a given subject
    scratch_folder = SCRATCH_FOLDER / id
    if not scratch_folder.exists():
        print('making scratch folder: ', scratch_folder)
        scratch_folder.mkdir(exist_ok=False)
    return scratch_folder

def make_raw(id):
    # Making a raw object from a .mat file and saving it as a .fif file 
    # in the scratch/session folder
    outfile = SCRATCH_FOLDER / id / RAW_FILENAME
    if outfile.exists() and not configs['overwrite']:
        print(f'{outfile} already exists. Skipping.')
        return None
    
    raws = []
    for session in range(1,N_SESSIONS+1):
        infile = RAW_FOLDER / id / str(session) / RAW_MAT_FILENAME

        print(f'Reading {infile}')
        # Read the .mat data file
        data = mat73.loadmat(infile)

        eeg = data['y'][1:-4,:] # removing unnecessary channels

        # Set first and last sample based on first and last trigger
        first_samp = np.where(eeg[32,:] != 0)[0][0]
        last_samp = np.where(eeg[32,:] == eeg[32,:].max())[0][0]

        first_conservative = first_samp - configs['t_before_first_trig']*SFREQ
        last_conservative = last_samp + configs['t_after_last_trig']*SFREQ

        # Crop data to first and last sample based on triggers
        print(f'First sample: {first_conservative}, last sample: {last_conservative}')
        print('Cropping data based on triggers...')
        print(f'data cut from {eeg.shape[1]} to {last_samp-first_samp} samples')
        eeg = eeg[:,first_conservative:last_conservative]

        # Set channel names
        ch_names = [f'CH {i}' for i in range(1,len(CH_POSITIONS)+1)]

        # Exchange channel names with positions
        ch_positions = [CH_POSITIONS.get(ch,ch) for ch in ch_names] + ['trigger']
        ch_types = ['eeg' for _ in ch_names] + ['stim']

        # Create info mne object
        info = mne.create_info(ch_positions, SFREQ, ch_types)
        info['subject_info'] = {'his_id':id} 

        # Make raw object 
        raw = mne.io.RawArray(eeg, info)
        raw = raw.apply_function(lambda x: x*1e-6) # convert to volts

        # Set channel positions
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        raws.append(raw)

    raw_full = mne.concatenate_raws(raws)
    
    # find breaks
    raw_full = find_breaks(raw_full)
    
    raw_full.save(outfile, overwrite=True)
    
def find_breaks(raw):
    events = mne.find_events(raw, stim_channel='trigger', verbose=False)

    durs = []
    starts = []
    labels = []
    break_number = 0
    for i in range(len(events)):
        ev = events[i]
        if ev[1] % N_TRIALS_PER_BREAK == 0 and ev[1] != 0:
            start = (events[i-1][0] / SFREQ) + 2.5
            end = (ev[0]/SFREQ) -0.5
            dur = end-start
            break_number += 1
            durs.append(dur)
            starts.append(start)
            labels.append(f'BAD_BREAK_{break_number}') 
    annotations = mne.Annotations(onset=starts, duration=durs, description=labels)
    raw.set_annotations(annotations)
    return raw

def read_and_prep_files(id):
    make_scratch(id)
    make_raw(id)