# Python
import cv2
import os

# Replace with your actual directory
directory = "../data/combined_data"

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_ANYDEPTH)
        depth = img.dtype
        print(f"{filename}: {depth}")