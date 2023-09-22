# Python
import cv2
import numpy as np

def read_data(img_mask_path, convert_to_16bit=True):
    img_path = img_mask_path[0]
    mask_path = img_mask_path[1]

    # Load images as 16-bit grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # check bit depth of image and convert if needed
    if convert_to_16bit:
        if img.dtype == np.uint8:
            img = img.astype(np.uint16) * 256
        elif img.dtype == np.uint16:
            pass
        else:
            raise TypeError("Image bit depth is not 8 or 16 bits")
    # convert and 16 bit images to 8 bit
    else:
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        else:
            raise TypeError("Image bit depth is not 8 or 16 bits")
        
    # Scale mask back to 8-bit(if needed)
    if mask.dtype == np.uint16:
        mask = (mask / 256).astype(np.uint8)
    elif mask.dtype == np.uint8:
        pass

    # Add an extra dimension to match the input shape for the model
    img = img[..., np.newaxis]
    mask = mask[..., np.newaxis]

    return img, mask

def resize_image(out_img, output_size):
    # Resize the image
    out_img = cv2.resize(out_img, output_size, interpolation=cv2.INTER_AREA)

    # Normalize the image to [0, 1]
    out_img = out_img / 65535.0  # 65535 is the maximum value for 16-bit grayscale

    return out_img