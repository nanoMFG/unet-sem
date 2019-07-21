from model import *
#from utils import *
import argparse
import sys
#import tensorflow as tf
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.preprocessing.image import ImageDataGenerator
#import keras.backend as K
#from keras.utils import multi_gpu_model
#
#if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--crop', action = 'store_true', default = True, help='Constructs batch using random crops.')
parser.add_argument('--augment', action = 'store_true', default = True, help='Constructs batch using augmentations.')
parser.add_argument('--max_crop', action = 'store_true', default = False, help='Crops using the maximum square size for each image. Crop size is ignored.')
parser.add_argument('--crop_size', default = 256, type = int, help='Size of cropped sample.')
parser.add_argument('--input_size', default = 256, type = int, help='Model input size. Cropped images will be rescaled to this size.')
parser.add_argument('--ngpu', default=1,type = int, help='Number of GPUs.')
parser.add_argument('--nepochs', default = 5, type = int, help='Number of epochs.')
parser.add_argument('--batch_size', default = 32, type = int, help='Number of samples per batch.')
parser.add_argument('--split', default = 0.1, type = float, help='Fraction of data to use for validation.')
parser.add_argument('--lr', default = 1e-4, type = float, help='Learning rate.')

kwargs = vars(parser.parse_args())

kwargs['crop_size'] = (kwargs['crop_size'],kwargs['crop_size'])
kwargs['input_size'] = (kwargs['input_size'],kwargs['input_size'])
print(kwargs)


run = TrainUNET(**kwargs)
run.train()
