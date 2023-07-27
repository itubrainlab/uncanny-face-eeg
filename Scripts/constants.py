from pathlib import Path
# Recording Constants
SFREQ = 256

# Channel positions
CH_POSITIONS = {
    'CH 1': 'Fp1',
    'CH 2': 'Fp2',
    'CH 3': 'Fz',
    'CH 4': 'FCz',
    'CH 5': 'CPz',
    'CH 6': 'F4',
    'CH 7': 'F8',
    'CH 8': 'FT8',
    'CH 9': 'FC4',
    'CH 10': 'C4',
    'CH 11': 'CP4',
    'CH 12': 'F7',
    'CH 13': 'F3',
    'CH 14': 'FC3',
    'CH 15': 'C3',
    'CH 16': 'CP3',
    'CH 17': 'FT7',
    'CH 18': 'T7',
    'CH 19': 'P3',
    'CH 20': 'TP7',
    'CH 21': 'P7',
    'CH 22': 'T8', 
    'CH 23': 'TP8',
    'CH 24': 'P8',
    'CH 25': 'P4',
    'CH 26': 'Pz',
    'CH 27': 'O1',
    'CH 28': 'Oz',
    'CH 29': 'O2',
    'CH 30': 'Cz',
    'CH 31': 'TP9',
    'CH 32': 'TP10',
}

# Paths
RAW_FOLDER = Path(__file__).parent.parent / 'Raw' 
SCRATCH_FOLDER = Path(__file__).parent.parent / 'Scratch'
RESULTS_FOLDER = Path(__file__).parent.parent / 'Results'

# Filenames
## RAW
LOGFILE_FILENAME = 'logfile.log'
RAW_MAT_FILENAME = 'eeg.mat'
BEHAVE_FILENAME  = 'behave.csv'

RAW_FILENAME = 'raw_eeg.fif'
RAW_PREP_FILENAME = 'prep_eeg.fif'

EPOCHS_FILENAME = 'prep_eeg_epo.fif'
ICA_FILENAME = 'eeg_ica.fif'

BAD_CHANNELS_FILENAME = 'bad_channels.txt'
ERP_FILENAME = 'ERP_data.csv'
TIME_TABLE_FILENAME = 'times.csv'

# Logfile
LOG_COLUMNS = ['TIME', 'LEVEL', 'MSG']
CONDITIONS =  ['animated', 'human', 'semi-realistic']
BEHAVE_CONDITIONS = ['inanimate', 'in-between', 'alive']

# Experiment
N_TRIALS_PER_BREAK = 20
N_SESSIONS = 2

# Subjects
SUBJECTS = [folder.name for folder in RAW_FOLDER.glob('*') if folder.is_dir()]
SUBJECTS.sort()
#SUBJECTS = ['a', 'b', 'c', 'd', 'e']

# Category mapping
def CATEGORY_MAPPING(img_id, output='str'):
    if output == 'str':
        if img_id <= 36:
            return 'animated'
        if img_id <= 72:
            return 'human'
        return 'semi-realistic'
    if output == 'int':
        if img_id <= 36:
            return 2
        if img_id <= 72:
            return 1
        return 0