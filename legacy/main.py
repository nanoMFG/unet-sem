#from model import *
#from utils import *
import argparse
import sys
import signal
#import tensorflow as tf
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.preprocessing.image import ImageDataGenerator
#import keras.backend as K
#from keras.utils import multi_gpu_model
#
#if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--crop', action = 'store_true', default = False, help='Constructs batch using random crops.')
parser.add_argument('--shuffle_data', action = 'store_true', default = False, help='Shuffles data paths to switch what is in test/train.')
parser.add_argument('--augment_after', default = 0, type = int, help='Start augmenting data ater specified epoch, inclusively.')
parser.add_argument('--augment', action = 'store_true', default = False, help='Constructs batch using augmentations.')
parser.add_argument('--max_crop', action = 'store_true', default = False, help='Crops using the maximum square size for each image. Crop size is ignored.')
parser.add_argument('--crop_size', default = 256, type = int, help='Size of cropped sample.')
parser.add_argument('--input_size', default = 256, type = int, help='Model input size. Cropped images will be rescaled to this size.')
parser.add_argument('--ngpu', default = 1, type = int, help='Number of GPUs.')
parser.add_argument('--nepochs', default = 5, type = int, help='Number of epochs.')
parser.add_argument('--batch_size', default = 32, type = int, help='Number of samples per batch.')
parser.add_argument('--split', default = 0.1, type = float, help='If float, fraction of data to use for validation. If integer, number of folds. If zero, train on all data (used for final model.')
parser.add_argument('--lr', default = 1e-4, type = float, help='Learning rate.')
parser.add_argument('--input_dir', default = 'data/', type = str, help='Directory to pull images from')

kwargs = vars(parser.parse_args())

kwargs['crop_size'] = (kwargs['crop_size'],kwargs['crop_size'])
kwargs['input_size'] = (kwargs['input_size'],kwargs['input_size'])
print(kwargs)

run = TrainUNET(**kwargs)

if kwargs['split']==0:
	run.trainAll()
elif kwargs['split'].is_integer():
	run.kFoldValidation(folds=int(kwargs['split']))
else:
	run.train()
