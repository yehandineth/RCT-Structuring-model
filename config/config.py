from os.path import abspath
from pathlib import Path

#Change these two variables when changing training dataset
DF = '20K_replaced_numbers'
VOCAB_SIZE = 68000

NUM_EPOCHS = 1

SEED = 42

TRANSITION_WEIGHT = 0.33

EMBEDDING_DIMENSIONS = 128

NUM_TOKENS = 56

CHARACTER_TOKEN_LENGTH = 296

CHARACTER_VOCAB_SIZE = 28

CHARACTER_EMBEDDING_DIMENSIONS = 32

BATCH_SIZE = 32

CLASS_NAMES = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

NUM_CLASSES = len(CLASS_NAMES)

############################### DIRECTORIES ##############################

#
MAIN_DIR = Path(abspath(__file__)).parent.parent
##
DATASET_DIR = MAIN_DIR.joinpath('datasets')
##
SAVES_DIR = MAIN_DIR.joinpath('saves')
###
MODEL_DIR = SAVES_DIR.joinpath('models')
###
WEIGHTS_DIR = SAVES_DIR.joinpath('weights')
##
TRAIN_DIR = MAIN_DIR.joinpath('training')
##
CALLBACKS_DIR = TRAIN_DIR.joinpath('callbacks')
##
EXPERIMENTS_DIR = MAIN_DIR.joinpath('experiments')
##
SERIALIZATION_DIR = MAIN_DIR.joinpath('serialization')
##
CHECKPOINTS_DIR = TRAIN_DIR.joinpath('checkpoints')

########################################################################

DATASET_PATHS = {
    '20K' : DATASET_DIR.joinpath("PubMed_20k_RCT"),
    '20K_replaced_numbers' : DATASET_DIR.joinpath("PubMed_20k_RCT_numbers_replaced_with_at_sign"),
    '200K' : DATASET_DIR.joinpath("PubMed_200k_RCT"),
    '200K_replaced_numbers' : DATASET_DIR.joinpath("PubMed_200k_RCT_numbers_replaced_with_at_sign"),
}