from model import *
from utils import *
import argparse
import sys
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import multi_gpu_model

class TrainUNET:
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    image_mask_paths = [("data/image<%d>.tif"%i,"data/image_mask<%d>.jpg"%i) for i in range(1,41)]
    def __init__(self,crop=True,augment=True,nepochs=5,batch_size=32,split=0.1,max_crop=False,crop_size=(256,256),input_size=(256,256),ngpu=1,lr=1e-4):
        num_images = len(image_mask_paths)
        self.test_paths = image_mask_paths[:int(split*num_images)]
        self.train_paths = image_mask_paths[int(split*num_images):]

        self.crop = crop
        self.augment = augment
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.split = split
        self.max_crop = max_crop
        self.crop_size = crop_size
        self.input_size = input_size
        self.ngpu = ngpu
        self.lr = lr

        self.model = unet(input_size=self.input_size+(1,))
        if self.ngpu > 1:
            self.model = multi_gpu_model(self.model,gpus=self.ngpu)
        self.model.compile(optimizer = Adam(lr = self.lr), loss = 'binary_crossentropy', metrics = ['accuracy'])

    def train(self):    
        for epoch in range(self.nepochs):
            shuffle(self.train_paths)
            for i, img_mask_path in enumerate(self.train_paths):
                img, mask = read_data(img_mask_path)
                aug_imgs, aug_masks = generate_batch(
                    img,
                    mask,
                    batch_size=self.batch_size,
                    random_crop_size=self.crop_size,
                    output_size=self.input_size,
                    crop = self.crop,
                    augment = self.augment,
                    aug_dict=data_gen_args,
                    max_crop = self.max_crop)
                loss = self.model.train_on_batch(aug_imgs,aug_masks)
                print("epoch: %d (%d/%d), %s: %s"%(epoch,i,len(self.train_paths),self.model.metrics_names,loss))

            test_loss = []
            for i, img_mask_path in enumerate(self.test_paths):
                img, mask = read_data(img_mask_path)
                out_imgs,out_masks = generate_batch(
                    img,
                    mask,
                    batch_size=self.batch_size,
                    random_crop_size=self.crop_size,
                    output_size=self.input_size,
                    crop = self.crop,
                    augment = False,
                    aug_dict=data_gen_args,
                    max_crop = self.max_crop)

                test_loss.append(self.model.test_on_batch(out_imgs,out_masks))
            test_loss = np.mean(np.array(test_loss),axis=0)
            print("epoch: %d, %s: %s"%(epoch,self.model.metrics_names,test_loss))

if ___name___ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', action = 'store_true', default = True, help='Constructs batch using random crops.')
    parser.add_argument('--augment', action = 'store_true', default = True, help='Constructs batch using augmentations.')
    parser.add_argument('--max_crop', action = 'store_true', default = False, help='Crops using the maximum square size for each image. Crop size is ignored.')
    parser.add_argument('--crop_size', nargs = 1, default = 256, type = int, help='Size of cropped sample.')
    parser.add_argument('--input_size', nargs = 1, default = 256, type = int, help='Model input size. Cropped images will be rescaled to this size.')
    parser.add_argument('--ngpu', nargs = 1, default = 1, type = int, help='Number of GPUs.')
    parser.add_argument('--nepochs', nargs = 1, default = 5, type = int, help='Number of epochs.')
    parser.add_argument('--batch_size', nargs = 1, default = 32, type = int, help='Number of samples per batch.')
    parser.add_argument('--split', nargs = 1, default = 0.1, type = float, help='Fraction of data to use for validation.')
    parser.add_argument('--lr', nargs = 1, default = 1e-4, type = float, help='Learning rate.')

    kwargs = vars(parser.parse_args(sys.argv))

    run = TrainUNET(**kwargs)
    run.train()
