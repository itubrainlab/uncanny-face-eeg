from pathlib import Path

### SUBJECTS IDENTIFICATION ###
SUBJECT = "test"
STOP_EARLY = False

# Screen info
FRAMERATE = 60

RESIZE_FACTOR = 2/3

# Fixation cross size
outer_size = 2
inner_size = 0.3

# File paths
WD = Path(__file__).parent.parent
IMAGE_FOLDER = WD / 'images' 
DATA_FOLDER = WD / 'data'
CATALOGUE_FILE = WD / 'image_catalogue.json'


# Colours
BACKGROUND = 153
FOREGROUND = 200

# Times 
TRIAL_TIME = 4
ANS_TIME = 2.5
STIM_TIME = 0.6
BREAKTIME = 10

SOA_MIN = 300
SOA_MAX = 600

EXP_LONG = 700
EXP_SHORT = 100

# Keys
ANS_KEYS = ['1','2','3']

# Logging
HEADER = ['trial_number', 'keypress', 'answering_time', 'image_id', 'exposure_time', 'SOA']

# Good images
BEST_STIM = [71,  64,  72,  47,  60,  58,  57,  50,  45,  48,  51,  53,  73,
             43,  65,  59,  62,  70,  52,  22,  96,   8, 108,  99,  35,  16,
             18,  17,  24,   2,  32, 103,  34,  33, 100,  84,  86,  13,  82,
             92,  12, 106,  83,  25,  20,  87,  29, 109,  97,  23,  37,  15,
             31, 107,  93,  19,  98,  85,  42, 102]
