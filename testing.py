import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

from Helpers.ImageHelper import Image_Helper

def cropImages():
    image_helper = Image_Helper()
    image_helper.img_dir = 'C:\\Users\\F5056756\\OneDrive - FRG\\7. Jurg Play\\data\\imgrec-potholes\\all_data\\all_data\\aBIvBVzRyDRhHaC'
    image_helper.read_ext = '.jpg'
    image_helper.read_image()
    plt.imshow(image_helper.image)
    plt.show()
    image_helper.crop_image()
    plt.imshow(image_helper.image)
    plt.show()

    image_helper.rotate_image(angles=[90, 180, 270])

    for img in image_helper.rotated_images:
        plt.imshow(img)


cropImages()
pass