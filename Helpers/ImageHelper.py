import cv2
import numpy as np
import imutils

from Helpers.PickleHelper import Pickle_Helper

pickler = Pickle_Helper()

class Image_Helper:

    def __init__(self):
        self.img_dir = None
        self.read_ext = None
        self.new_dir = ''
        self.image = None
        self.reshaped_image = None
        self.rotated_images = []
        self.reshaped_rotated_images = []

    def read_image(self):
        file_name = self.img_dir + self.read_ext
        try:
            img_array = cv2.imread(file_name)
            res = cv2.resize(img_array, (800, 600), interpolation=cv2.INTER_CUBIC)
            self.image = res
        except Exception as e:
            print(e)
            print('Problem reading image: ' + file_name)

    def crop_image(self):
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[0, 200], [800, 200], [0, 400], [800, 400]])
        pts2 = np.float32([[0, 0], [600, 0], [0, 200], [600, 200]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(self.image, M, (600, 200))
        self.image = dst

    def rotate_image(self, angles):
        for angle in angles:
            dst = imutils.rotate_bound(self.image, angle)
            self.rotated_images.append(dst)

    @staticmethod
    def reshape_n_2(img):
        reshaped_image = np.array(img).reshape(1, -1)
        return reshaped_image

    def reshape_image(self):
        self.reshaped_image = self.reshape_n_2(self.image)

    def reshape_rotated_images(self):
        for rotated_image in self.rotated_images:
            self.reshaped_rotated_images.append(self.reshape_n_2(rotated_image))

    def write_image(self):
        pickler.write_pickle(self.image, self.new_dir)

    def write_reshaped_image(self):
        pickler.write_pickle(self.reshaped_image, self.new_dir)

    def write_rotated_images(self):
        for i, image in enumerate(self.rotated_images):
            file_name = self.new_dir + "_" + str(i+1)
            pickler.write_pickle(image, file_name)

    def write_reshaped_rotated_images(self):
        for i, image in enumerate(self.reshaped_rotated_images):
            file_name = self.new_dir + "_" + str(i+1)
            pickler.write_pickle(image, file_name)
