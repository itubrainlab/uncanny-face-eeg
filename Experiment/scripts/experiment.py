import time

from psychopy import visual, event, core, logging
from random import shuffle, randrange
import json, datetime, csv

from constants import *
from fix import make_fix
from trigger import send_trigger

def clean_quit():
    # close data file
    win.close()
    core.quit()
    raise KeyboardInterrupt

def sec_to_frames(t):
    return round(t * FRAMERATE)


def make_image_catalogue():
    from PIL import Image

    imageid = 0
    catalogue = {}

    for cat in IMAGE_FOLDER.glob('*'):
        if cat.is_dir():
            catname = cat.name
            catalogue[catname + '_first'] = imageid + 1
            for img in cat.glob('*.png'):
                imageid += 1
                imagefile = Image.open(img)
                width = imagefile.width
                height = imagefile.height
                catalogue[str(imageid)] = {
                    'path': str(img),
                    'cat': catname,
                    'size': (width, height)
                }
            catalogue[catname + '_last'] = imageid
    catalogue['last'] = imageid
    # Make the json file
    json_object = json.dumps(catalogue, indent=4)
    with open(CATALOGUE_FILE, "w") as outfile:
        outfile.write(json_object)


def make_trials():
    trials = []
    for int_id in BEST_STIM:
        strid = str(int_id)
        image_id = strid
        category = catalogue[strid]['cat']
        img_file = catalogue[strid]['path']
        img_size = catalogue[strid]['size']

        for exp_time in [EXP_SHORT, EXP_LONG]:
            jitter = randrange(-20, 20)
            real_exp_time = exp_time + jitter
            new_trial = Trial(category=category,
                              image_id=image_id,
                              img_file=img_file,
                              img_size=img_size,
                              exp_time=real_exp_time)
            trials.append(new_trial)
    for int_id in BEST_STIM:
        strid = str(int_id)
        image_id = strid
        category = catalogue[strid]['cat']
        img_file = catalogue[strid]['path']
        img_size = catalogue[strid]['size']

        for exp_time in [EXP_SHORT, EXP_LONG]:
            jitter = randrange(-20, 20)
            real_exp_time = exp_time + jitter
            new_trial = Trial(category=category,
                              image_id=image_id,
                              img_file=img_file,
                              img_size=img_size,
                              exp_time=real_exp_time)
            trials.append(new_trial)
    shuffle(trials)
    return trials


class Trial:

    def __init__(self, category, image_id, img_file, img_size, exp_time):
        self.category = category
        self.image_id = image_id
        self.img_file = img_file
        self.img_size = img_size
        self.exp_time = exp_time / 1000  # ms to sec

    def resize(self, img_size, factor):
        width = round(img_size[0] * factor)
        height = round(img_size[1] * factor)
        return (width, height)

    def prepare(self):
        new_size = self.resize(self.img_size, factor=RESIZE_FACTOR)
        self.stim = visual.ImageStim(win, image=self.img_file, size=new_size)
        self.SOA = randrange(SOA_MIN, SOA_MAX) / 1000  # ms to sec


### INITIALISATION ###
# Catalogue
if not CATALOGUE_FILE.is_file():
    make_image_catalogue()
with open(CATALOGUE_FILE, "r") as file:
    catalogue = json.load(file)

# global keys
event.globalKeys.add(key='q', func=core.quit, name='Quit')

# Window
win = visual.Window([1920, 1080], allowGUI=True, monitor='testMonitor', units='pix',
                    color=BACKGROUND, colorSpace='rgb255', fullscr=True)


# Trial list
trial_list = make_trials()
ans = []

# Stim setups - fix and text
fixation = make_fix(win)
rating = visual.TextStim(win, text='Please rate the image')

# Break text
break_stim = visual.TextStim(win, text='break')


# Before starting
intro = visual.TextStim(win, text= 'Welcome to the experiment!\n\n'
                                   'Here is what you will encounter in the experiment:\n'
                                       '1) fixation cross - please, just look at the centre of the cross,\n'
                                       '2) faces and faces!,\n'
                                       '3) rating the face - press "1" for low animacy, "2" for neutral animacy, "3" for high animacy - get prepared!'
                                       '    You have 2 seconds to answer! \n\n'
                                       'The experiment will last 15 minutes, but there will be breaks of 10 seconds every 20 trials\n'
                                       '\n\nPress any button to procede', wrapWidth=1000)
intro.draw()
win.flip()
event.waitKeys()

test = visual.TextStim(win, text='To get familiar with the experiment procedure, we will run 3 test trials.\nPlease follow the instructions on the screen\n\n'
                                'Press any button to procede', wrapWidth=1000)
test.draw()
win.flip()
event.waitKeys()

test.setText("This is the fixation cross, please keep your eyes on the centre during the experiment.\n\n"
             "The stimulus will appear when you press a button, the time it shows is very short, this is intentional.")
test.setPos((0, 300))
fixation.draw()
test.draw()
win.flip()
event.waitKeys()

strid = str(1)
image_id = strid
category = catalogue[strid]['cat']
img_file = catalogue[strid]['path']
img_size = catalogue[strid]['size']
test_trial = Trial(category=category,
                   image_id=image_id,
                   img_file=img_file,
                   img_size=img_size,
                   exp_time=100)
fixation.draw()
test_trial.prepare()
test_trial.stim.draw()
win.flip()
time.sleep(0.7)

test.setText("Please rate the image 'very animate' (3)")
fixation.draw()
test.draw()
win.flip()
event.waitKeys(keyList=['3'])

test_trial = Trial(category=category,
                   image_id=image_id,
                   img_file=img_file,
                   img_size=img_size,
                   exp_time=700)
fixation.draw()
test_trial.prepare()
test_trial.stim.draw()
win.flip()
time.sleep(0.1)

test.setText("Please rate the image 'low animacy' (1)")
fixation.draw()
test.draw()
win.flip()
event.waitKeys(keyList=['1'])

test_trial = Trial(category=category,
                   image_id=image_id,
                   img_file=img_file,
                   img_size=img_size,
                   exp_time=700)
fixation.draw()
test_trial.prepare()
test_trial.stim.draw()
win.flip()
time.sleep(0.7)

test.setText("Please rate the image 'medium animacy' (2)")
fixation.draw()
test.draw()
win.flip()
event.waitKeys(keyList=['2'])

start = visual.TextStim(win, text='This is the beginning of the real experiment, please ask questions now if you have any.\n\n'
                                  'Press any button to start')
start.draw()
win.flip()
event.waitKeys()

# Logging
start_time = datetime.datetime.now()
float_stamp = start_time.timestamp()
str_stamp = start_time.strftime("%Y-%m-%d-%H-%M-%S")

info_logger = logging.LogFile(f=f"logg\\logfile_{SUBJECT}_{str_stamp}", level=logging.EXP, filemode='w')

# Init datafile
datafile = DATA_FOLDER / f"data_{SUBJECT}_{str_stamp}.csv"
with open(datafile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

clock = core.Clock()

FRAMERATE = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)

for num_trial, trial in enumerate(trial_list, start=1):
    trial.prepare()
    stim = trial.stim
    SOA_frame = sec_to_frames(trial.SOA)
    stim_frame = sec_to_frames(trial.exp_time)
    even_frame = sec_to_frames(STIM_TIME - trial.exp_time)
    answering_frame = sec_to_frames(ANS_TIME)

    event.clearEvents()
    ans = []
    clock.reset()
    # SOA
    win.logOnFlip(level=logging.EXP, msg='TRIAL-BEGIN_' + str(num_trial))
    for frame in range(SOA_frame):
        fixation.draw()
        win.flip()
    # Stim
    win.logOnFlip(level=logging.EXP, msg='STIM_' + str(num_trial))
    send_trigger(num_trial, trial.image_id, trial.exp_time, float_stamp)
    for frame in range(stim_frame):
        stim.draw()
        win.flip()

    # Evening
    for frame in range(even_frame):
        win.flip()
    # Answering
    win.logOnFlip(level=logging.EXP, msg='ANSWER_' + str(num_trial))
    for frame in range(answering_frame):
        rating.draw()
        win.flip()
        ans = event.getKeys(keyList=ANS_KEYS, timeStamped=clock)
        if len(ans) != 0:  # Check if answered
            data = [num_trial, ans[0][0], ans[0][1], trial.image_id, trial.exp_time, trial.SOA]
            break
        elif frame == answering_frame - 1:
            data = [num_trial, 'MISSED', 0, trial.image_id, trial.exp_time, trial.SOA]

    # write to datafile
    with open(datafile, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    # Wait for the trial to finish
    win.logOnFlip(level=logging.EXP, msg='REST_' + str(num_trial))
    leftover = TRIAL_TIME - clock.getTime()
    for frame in range(sec_to_frames(leftover)):
        fixation.draw()
        win.flip()

    if num_trial % 20 == 0:
        for frame in range(sec_to_frames(BREAKTIME)):
            timer = round(10 - frame//FRAMERATE)
            break_text = f"{timer}\n\n Break time - rest your eyes"
            break_stim.setText(break_text)
            break_stim.draw()
            win.flip()

    if num_trial > 1 and STOP_EARLY: # Stop for testing
        break


    

thanks = visual.TextStim(win, text='Experiment has ended.\nThank you for your participation!')
thanks.draw()
win.flip()
event.waitKeys()

