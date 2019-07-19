import numpy as np
import cv2
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

def read_data(img_mask_path):
    img_path = img_mask_path[0]
    mask_path = img_mask_path[1]
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = img[...,np.newaxis]
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    mask = mask[...,np.newaxis]

    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask

def random_crop(img, mask, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]

def augment_input(img,mask,aug_dict,batch_size=32,random_crop_size=(256,256)):
    LX,LY,LZ = img.shape
    image_datagen = ImageDataGenerator(**aug_dict)

    out_imgs = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    out_masks = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    for b in range(batch_size):
        seed = np.random.randint(1e9)
        crop_img, crop_mask = random_crop(img,mask,random_crop_size)
        out_imgs[b,...] = image_datagen.random_transform(crop_img,seed=seed)
        out_masks[b,...] = image_datagen.random_transform(crop_mask,seed=seed)

    return out_imgs, out_masks
