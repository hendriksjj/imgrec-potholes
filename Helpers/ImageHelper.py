import cv2
import numpy as np

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

    def rotate_image(self, angles):
        rows, cols = self.image.shape[:2]
        # cols-1 and rows-1 are the coordinate limits.
        for angle in angles:
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
            dst = cv2.warpAffine(self.image, M, (cols, rows))
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
