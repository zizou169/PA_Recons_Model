'''
TODO:
    Configurations for UNet
    Hyper parameters
'''

__author__ = 'ZY'

# Main thread controllers
TRAIN_FLAG = True # if training
TEST_FLAG = False # if test
MODEL_SAVE_FLAG = True # if saving model
MODEL_LOAD_FLAG = False # if load training checkpoint
TRAINING_VISUAL_FLAG = False # if activate training visualization
TENSORBOARD_FLAG = True # if use Tensorboard

# directories
# local
ROOT_PATH = 'C:/Users/Tri Vu/OneDrive - Duke University/PI Lab/Limited-View DL/'
DATA_PATH = ROOT_PATH + 'Data Generation/'
MODEL_LOAD_PATH = ROOT_PATH + 'models/PA_UNet_Yuan_090918.h5'
MODEL_SAVE_PATH = ROOT_PATH + 'models/PA_UNet_Yuan_090918.h5'
TRAIN_DATA_PATH = DATA_PATH + '2D homo high disc line/'
TEST_DATA_PATH = DATA_PATH + 'test_simu/'
PRED_DATA_PATH = DATA_PATH + 'pred_invivo_vessel/'
LOG_PATH = ROOT_PATH + 'logs/'

# input data
INPUT_SIZE = 128
INPUT_CHANNEL = 1   # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 1

# network structure
FILTER_NUM = 32 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 2 # size of upsampling filters

# network hyper-parameter
DROPOUT_RATE = 0.4
BATCH_NORM_FLAG = True

# training data
TRAINING_SIZE = 600
TRAINING_START = 1
BATCH_SIZE = 40
EPOCH = 1
VALIDATION_SPLIT = 0.30

# test data
TEST_SIZE = 2
TEST_START = 601
