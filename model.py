from utils import *
from keras.models import *
from keras.layers import *
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.preprocessing.image import ImageDataGenerator
#import keras.backend as K
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from sklearn.model_selection import KFold
import time
import os
import numpy as np

K.set_floatx('float32')

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


class TrainUNET:
    def __init__(self,crop=True,
                    augment=True,
                    nepochs=5,
                    batch_size=32,
                    split=0.1,
                    max_crop=False,
                    crop_size=(256,256),
                    input_size=(256,256),
                    ngpu=1,
                    lr=1e-4,
                    shuffle_data=False,
                    augment_after=0,
                    output_dir=None,
                    input_dir=None):

        self.data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range=[0.2,1.8],
                    fill_mode='nearest')

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
        self.shuffle_data = shuffle_data
        self.augment_after = augment_after

        if input_dir is not None:
            # Automatically scan directory for all images, need to add some
            # error checking to make sure everything exists
            self.image_mask_paths = []
            # https://stackoverflow.com/a/3964691
            for file in os.listdir(input_dir):
                if file.endswith('.tif'):
                    #DO NOT LEAVE THESE PRINT STATEMENTS, THEY NEED TO BE LOGGED INSTEAD
                    image = os.path.join(input_dir, file)
                    num = int(file.split('<')[1].split('>')[0])
                    mask = os.path.join(input_dir, 'image_mask<{}>.jpg'.format(num))
                    print('Loading image path', image, 'and mask path', mask)
                    self.image_mask_paths.append((image, mask))
        else:
            self.image_mask_paths = [("data/image<%d>.tif"%i,"data/image_mask<%d>.jpg"%i) for i in range(1,41)]

        if self.shuffle_data:
            shuffle(self.image_mask_paths)
        num_images = len(self.image_mask_paths)
        self.test_paths = self.image_mask_paths[:int(self.split*num_images)]
        self.train_paths = self.image_mask_paths[int(self.split*num_images):]

        self.instantiate_model()

    def instantiate_model(self):
        self.serial_model = unet(input_size=self.input_size+(1,))
        if self.ngpu > 1:
            self.model = multi_gpu_model(self.serial_model,gpus=self.ngpu)
        else:
            self.model = self.serial_model

        optimizer = RMSprop
        # optimizer = Adam
        self.model.compile(optimizer = optimizer(lr = self.lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
        # plot_model(self.model, to_file='model.png',show_shapes=True)

    def kFoldValidation(self,folds=10,random_state=9999):
        kf = KFold(n_splits=folds,shuffle=True,random_state=random_state)
        k = 0
        acc_list = []
        loss_list = []
        for train_idxs, test_idxs in kf.split(self.image_mask_paths):
            self.instantiate_model()

            self.train_paths = [self.image_mask_paths[i] for i in train_idxs]
            self.test_paths = [self.image_mask_paths[i] for i in test_idxs]

            test_results = self.train(save_dir="FOLD_%02d"%k)
            loss_list.append(test_results['loss'])
            acc_list.append(test_results['acc'])

            k+=1
        print("[KFOLD_METRICS] ACC: %s +/- %s LOSS: %s +/- %s"%(np.mean(acc_list),np.std(acc_list),np.mean(loss_list),np.std(loss_list)))

    def trainAll(self):
        self.train_paths = self.image_mask_paths
        append_to_log("[ALL_DATA]",directory="AllData")
        self.train(test=False,save_dir="AllData")

    def train(self,test=True,save_dir='output',out_file='out.log'):
        for key, value in self.__dict__.items():
            if key not in ['test_paths','train_paths'] and not key.startswith("__"):
                append_to_log("%s: %s"%(key,value),directory=save_dir,filename=out_file)

        best_acc = 0

        #this should be moved to utils eventually
        num_imgs = len(self.train_paths) * self.batch_size
        imgs = np.zeros((num_imgs, self.input_size[0], self.input_size[1], 1))
        masks = np.zeros((num_imgs, self.input_size[0], self.input_size[1], 1))
        i = 0
        for img_path, img_mask_path in enumerate(self.train_paths):
            img, mask = read_data(img_mask_path)
            aug_imgs, aug_masks = generate_batch(
                        img,
                        mask,
                        batch_size=self.batch_size,
                        random_crop_size=self.crop_size,
                        output_size=self.input_size,
                        crop = self.crop,
                        augment = self.augment,
                        aug_dict=self.data_gen_args,
                        max_crop = self.max_crop)
            #Can likely convert this to use slicing instead
            for aug_img in aug_imgs:
                imgs[i] = aug_img
            for aug_mask in aug_masks:
                masks[i] = aug_mask
            i += 1

        num_test_imgs = len(self.test_paths) * self.batch_size
        test_imgs = np.zeros((num_test_imgs, self.input_size[0], self.input_size[1], 1))
        test_masks = np.zeros((num_test_imgs, self.input_size[0], self.input_size[1], 1))
        i = 0
        for img_path, img_mask_path in enumerate(self.test_paths):
            img, mask = read_data(img_mask_path)
            aug_imgs, aug_masks = generate_batch(
                        img,
                        mask,
                        batch_size=self.batch_size,
                        random_crop_size=self.crop_size,
                        output_size=self.input_size,
                        crop = self.crop,
                        augment = self.augment,
                        aug_dict=self.data_gen_args,
                        max_crop = self.max_crop)
            #Can likely convert this to use slicing instead
            for aug_img in aug_imgs:
                test_imgs[i] = aug_img
            for aug_mask in aug_masks:
                test_masks[i] = aug_mask
            i += 1

        tensorboard_cb = TensorBoard('logs/fit/', histogram_freq=1)
        self.model.fit(imgs, masks,
            batch_size=self.batch_size,
            epochs=self.nepochs,
            shuffle=True,
            validation_data=(test_imgs, test_masks),
            callbacks=[tensorboard_cb])
        # for epoch in range(self.nepochs):
        #     shuffle(self.train_paths)
        #     for i, img_mask_path in enumerate(self.train_paths):
        #         img, mask = read_data(img_mask_path)
        #
        #         if epoch >= self.augment_after:
        #             augment = self.augment
        #             crop = self.crop
        #         else:
        #             augment = False
        #             crop = False
        #
        #         aug_imgs, aug_masks = generate_batch(
        #             img,
        #             mask,
        #             batch_size=self.batch_size,
        #             random_crop_size=self.crop_size,
        #             output_size=self.input_size,
        #             crop = self.crop,
        #             augment = augment,
        #             aug_dict=self.data_gen_args,
        #             max_crop = self.max_crop)
        #         loss = self.model.train_on_batch(aug_imgs,aug_masks)
        #         append_to_log(
        #             "[TRAIN] epoch: %d (%d/%d), %s: %s %s"%(epoch,i,len(self.train_paths),self.model.metrics_names,loss[0],loss[1]),
        #             directory = save_dir,
        #             filename = out_file
        #             )
        #
        #     if test:
        #         test_loss = []
        #         for i, img_mask_path in enumerate(self.test_paths):
        #             img, mask = read_data(img_mask_path)
        #             out_imgs,out_masks = generate_batch(
        #                 img,
        #                 mask,
        #                 output_size=self.input_size,
        #                 batch_size=1,
        #                 crop = False,
        #                 augment = False,)
        #
        #             prediction = self.model.predict_on_batch(out_imgs)
        #             if epoch % 20 == 0 or epoch == self.nepochs-1:
        #                 save_output(out_imgs[0,...],out_masks[0,...],prediction[0,...],index=i,epoch=epoch,directory=save_dir)
        #             test_loss.append(self.model.test_on_batch(out_imgs,out_masks))
        #         test_loss = np.mean(np.array(test_loss),axis=0)
        #         if test_loss[1]>best_acc:
        #             save_model_unet(self.serial_model,epoch,test_loss[1],directory=save_dir)
        #         append_to_log(
        #             "[TEST] epoch: %d, %s: %s %s"%(epoch,self.model.metrics_names,test_loss[0],test_loss[1]),
        #             directory = save_dir,
        #             filename = out_file
        #             )
        #
        #         if epoch == self.nepochs-1:
        #             save_model_unet(self.serial_model,epoch,test_loss[1],directory=save_dir)
        #             return dict(zip(self.model.metrics_names,test_loss.tolist()))
        #
        #     if not test and epoch == self.nepochs-1:
        #         save_model_unet(self.serial_model,epoch,-1,directory=save_dir)
