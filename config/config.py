from os.path import abspath
from pathlib import Path

#Change each experiment
DF = '20K_replaced_numbers'
NAME = 'testing'

VOCAB_SIZE = 68000 if DF=='20K_replaced_numbers' else 331000

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
TRAIN_DIR = MAIN_DIR.joinpath('training')
##
TEST_DIR = MAIN_DIR.joinpath('testing')
##
CALLBACKS_DIR = TRAIN_DIR.joinpath('callbacks')
##
TRAINING_DATA_DIR = TRAIN_DIR.joinpath('training_data')
##
SERIALIZATION_DIR = MAIN_DIR.joinpath('serialization')
##
CHECKPOINTS_DIR = TRAIN_DIR.joinpath('checkpoints')

########################################################################

DATASET_PATHS = {
    '20K_replaced_numbers' : DATASET_DIR.joinpath("PubMed_20k_RCT_numbers_replaced_with_at_sign"),
    '200K' : DATASET_DIR.joinpath("PubMed_200k_RCT")
}