# Python
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ... rest of your code ...
import numpy as np
import cv2
from unetSem.data.loader import read_data, resize_image

# def test_read_data():
#     # Replace with your actual paths
#     img_path = "data/combined_data/image_0001.tif"
#     mask_path = "data/combined_data/image_mask_0001.png"

#     img, mask = read_data([img_path, mask_path])

#     # Check that the images have the correct shape
#     assert img.ndim == 3
#     assert mask.ndim == 3

#     # Check that the images have the correct dtype
#     assert img.dtype in [np.uint8, np.uint16]
#     assert mask.dtype == np.uint8
def test_read_data_8bit():
    # Path to your 8-bit image and its mask
    img_path = "data/combined_data/image_0001.tif"
    mask_path = "data/combined_data/image_mask_0001.png"

    img, mask = read_data([img_path, mask_path], convert_to_16bit=False)

    # Check that the images have the correct shape
    assert img.ndim == 3
    assert mask.ndim == 3

    # Check that the images have the correct dtype
    assert img.dtype == np.uint8
    assert mask.dtype == np.uint8

def test_read_data_16bit():
    # Path to your 16-bit image and its mask
    img_path = "data/combined_data/image_0003.tif"
    mask_path = "data/combined_data/image_mask_0003.png"

    img, mask = read_data([img_path, mask_path], convert_to_16bit=True)

    # Check that the images have the correct shape
    assert img.ndim == 3
    assert mask.ndim == 3

    # Check that the images have the correct dtype
    assert img.dtype == np.uint16
    assert mask.dtype == np.uint8
def test_resize_image():
    # Create a dummy image
    img = np.zeros((100, 100), dtype=np.uint16)

    # Resize the image
    resized_img = resize_image(img, (50, 50))

    # Check that the image has been resized correctly
    assert resized_img.shape == (50, 50)

    # Check that the image values are in the correct range
    assert resized_img.min() >= 0.0
    assert resized_img.max() <= 1.0