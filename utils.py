import numpy as np
import cv2
from PIL import Image
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

def read_data(img_mask_path):
    img_path = img_mask_path[0]
    mask_path = img_mask_path[1]
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = img[...,np.newaxis]
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    mask = mask[...,np.newaxis]

    return img, mask

def random_crop(img, mask, random_crop_size=(256,256), max_crop=False):
    height, width = img.shape[0], img.shape[1]
    if max_crop:
        dx = dy = min(height,width)
    else:
        dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]

def random_augment(img, mask, output_size, aug_dict):
    image_datagen = ImageDataGenerator(**aug_dict)
    seed = np.random.randint(1e9)

    if img.shape != output_size:
        out_img = Image.fromarray(img[...,0])
        out_img.resize(output_size)
        out_mask = Image.fromarray(mask[...,0])
        out_mask.resize(output_size)

        out_img = np,array(out_img)[...,np.newaxis]
        out_mask = np,array(out_mask)[...,np.newaxis]

    out_img = image_datagen.random_transform(img,seed=seed)
    out_mask = image_datagen.random_transform(mask,seed=seed)

    return out_img, out_mask

def generate_batch(img, mask, batch_size=32, random_crop_size=(256,256), output_size=(256,256), crop = True, augment = True, aug_dict = {}, max_crop = False):
    LX,LY,LZ = img.shape

    out_imgs = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    out_masks = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    for b in range(batch_size):
        seed = np.random.randint(1e9)
        if crop:
            out_img, out_mask = random_crop(img,mask,random_crop_size)
        if augment:
            out_img, out_mask = random_augment(img,mask,output_size,aug_dict)

        out_img = out_img / 255
        out_mask = out_mask / 255
        out_mask[mask > 0.5] = 1
        out_mask[mask <= 0.5] = 0

        out_imgs[b,...] = out_img
        out_masks[b,...] = out_mask

    return out_imgs, out_masks    

def augment_input(img,mask,aug_dict,batch_size=32,random_crop_size=(256,256),only_crop=False):
    LX,LY,LZ = img.shape
    image_datagen = ImageDataGenerator(**aug_dict)

    out_imgs = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    out_masks = np.zeros((batch_size,random_crop_size[0],random_crop_size[1],1))
    for b in range(batch_size):
        seed = np.random.randint(1e9)
        crop_img, crop_mask = random_crop(img,mask,random_crop_size)
        if not only_crop:
            out_imgs[b,...] = image_datagen.random_transform(crop_img,seed=seed)
            out_masks[b,...] = image_datagen.random_transform(crop_mask,seed=seed)
        else:
            out_imgs[b,...] = crop_img
            out_masks[b,...] = crop_mask

    return out_imgs, out_masks