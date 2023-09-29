import sys

from read import read_and_prep_files
from preprocess import preprocess
from segmentation import segment
from analysis import analyse
from plotting import ss_plot, group_plot

from constants import SUBJECTS
from configs import configs, make_config_file, exp_folder

def full_run(subject):
    print(f'\n\nProcessing subject {subject}...')

    if configs['reading']:
        print('\nReading data...')
        read_and_prep_files(subject)

    if configs['preprocessing']:
        print('\nPreprocessing...')
        preprocess(subject)

    if configs['segmentation']:
        print('\nSegmenting...')
        segment(subject)

    if configs['analysis']:
        print('\nAnalyzing...')
        analyse(subject)

    if configs['ss_plot']:
        print('\nSingle Subject Plotting...')
        ss_plot(subject)
    

def create_experiment_folder():

    exp_folder.mkdir()
    
    # Create config file
    make_config_file(exp_folder)

    # Create folder for plots
    plotsfolder = exp_folder / 'plots'
    plotsfolder.mkdir()

def main():
    # Setup
    
    create_experiment_folder()
    print(f'Running experiment in folder: {exp_folder.name}')
    sys.stdout = open(exp_folder / 'log.txt', 'w')

    # Run
    for s in SUBJECTS:
        full_run(s)

    # Plot group results
    if configs['group_plot']:
        print('\nGroup plotting...')
        group_plot()

if __name__ == '__main__':
    main()