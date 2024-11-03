# import the necessary packages
import torch
import os
# base path of the dataset
#DATASET_PATH = os.path.join("dataset", "train")
# define the path to the images and masks dataset
# DATASET_PATH = os.path.join("/Users/jnaiman/Downloads/tmp/wormfindr/wormOntologyNSF/testSaltData", "train")
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
# MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# JPN -- updates for bossdb
IMAGE_DATASET_URI = "bossdb://white1986/n2u/em"
SEGMASK_DATASET_URI = "bossdb://white1986/n2u/seg"

# check for her eor others
if '/Users/jnaiman' in os.getcwd():
    BASE_OUTPUT = "/Users/jnaiman/Downloads/tmp/wormfindr/wormOntologyNSF/multiClassWorm256"
else:
    BASE_OUTPUT = "gdrive/MyDrive/Grants/WormFindr/multiClassWorm"

# how many test/training centroids?
NCENTROID_TRAIN = 100
NCENTROID_TEST = 100

NUM_CLASSES = 100 # assume we are just segmenting and not caring about kind

# not a uniform dataset
ZMIN,ZMAX = 143,159
YMIN,YMAX = 3661,4685
XMIN,XMAX = 5903,6927

# this is a WHOLE THING
ZSHIFT = 2

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
#INPUT_IMAGE_WIDTH = 128

INPUT_IMAGE_HEIGHT = INPUT_IMAGE_WIDTH # 128 -- JPN, these need to be the same

# ------ end updates for bossdb ------
# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1 # grayscale
#NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory

# define the path to the output serialized model, model training
# plot, and testing image paths
#MODEL_PATH = '/Users/jnaiman/Downloads/tmp/wormfindr/wormOntologyNSF/multiClassWorm/unet_model.pth'


MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_model.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# for plotting images
EVAL_IMG_PATHS = os.path.sep.join([BASE_OUTPUT, "eval_images/"])
