from datetime import datetime
from constants import RESULTS_FOLDER

#id = len(list(RESULTS_FOLDER.glob('*')))
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

configs = {

    'exp_name' : 'all prep',
    'timestamp' : timestamp,
    'overwrite' : 1,

    ### PARTS TO RUN ###
    'reading'       : 0,
    'preprocessing' : 0,
    'segmentation'  : 0,
    'analysis'      : 1,
    'ss_plot'       : 1,
    'group_plot'    : 0,

    # Preprocessing parts
    'run_ICA'   : 1,
    'run_ransac': 1,

    # Analysis parts
    'GFP' : 0,
    'ERP' : 0,
        'statistics' : 0,
    'EVK' : 0,
    'DISS': 0,
    'DECO': 0,
    'GFP_TEST': 0,
    'N400_test': 1,
    'N170_test': 0,

    # Plotting
    'violins' : 0,
    'evokeds' : 1,
    'compare_evokeds' : 1,
    'compare_topos' : 1,
    'plot_behaviour' : 0,
    'group_erp' : 0,

    ## Settings
    # Reading
    't_before_first_trig': 1,
    't_after_last_trig': 4,
    'break_plots': 0,

    # Epoching
    'e_tmin': -0.2,
    'e_tmax': 1,
    'e_baseline': (-0.2, 0),

    'tshift': -0.06,

    # Filtering
    'h_freq': 35,
    'l_freq': 1,

    # Reject epochs
    'reject': {'eeg': 100e-6}, # microvolts
    'reject_blink': {'eeg': 1000e-6}, # microvolts

    # Bad channels
    'bad_tstep' : 1,

    # ICA
    'ica_variances_explained': 0.99,
    'ica_method': 'infomax',
    'segmentation_tstep': 4,
    'random_state': 42,
    
    # Analysis
    'components': ['N170', 'N400'],#'P100', 'N170', 'P300', 'LPP'
    'img_avg': 1,
    'DISS_permutations' : 1000,
    'custom_times' : 1,
    'on_behave' : 0,
    'earlylate' : 0,
    'on_diff' : 0,
    'include_short' : 1,
    'standard_ref' : None, #None,['TP8'], 'average'

    'P100' : {
        'ch_picks': ['O2','Oz','O1'],
        'ref': 'average',
        'mode': 'pos',
        'times': (0.070,0.160),
        },
    'N170' : {
        'ch_picks': ['P8'],
        'ref': 'average',
        'mode': 'neg',
        'times' : (0.110,0.220),
        },
    'P300' : {
        'ch_picks': ['Pz'],
        'ref': None,
        'mode': 'pos',
        'times' : (0.200,0.600),
        },
    'N400' : {
        'ch_picks': ['CPz','CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4'],#['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],#, #
        'ref': None,
        'mode': 'neg',
        'times' : (0.45,0.6),
        },
    'LPP' : {
        'ch_picks': ['CP4','CP3','Pz'],#['CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4'], 'eeg'
        'ref': None,
        'mode': 'pos',
        'times' : (0.400,0.7),
    }
}

def make_config_file(exp_folder):
    config_file = exp_folder / 'config.txt'
    with open(config_file, 'w') as f:
        for key, value in configs.items():
            f.write(f'{key}: {value}\n\n')

exp_folder = RESULTS_FOLDER / f'{configs["exp_name"]}_{timestamp}'