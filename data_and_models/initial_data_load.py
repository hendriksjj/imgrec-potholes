import cv2
import shutil

import os
from Helpers.CsvHelper import CSV_Helper
from Helpers.ImageHelper import Image_Helper
from Helpers.PickleHelper import Pickle_Helper

pickler = Pickle_Helper()

def create_data_directories():
    # Creating the relevant directories
    data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data', 'imgrec-potholes')

    model_data_dir = os.path.join(data_directory, 'model_data')
    shutil.rmtree(model_data_dir)
    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)

    train_data_dir = os.path.join(model_data_dir, 'train')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    validate_data_dir = os.path.join(model_data_dir, 'validate')
    if not os.path.exists(validate_data_dir):
        os.makedirs(validate_data_dir)

    test_data_dir = os.path.join(model_data_dir, 'test')
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    submission_data_dir = os.path.join(data_directory, 'submission')
    return data_directory, model_data_dir, train_data_dir, validate_data_dir, test_data_dir, submission_data_dir

def initial_data_load():
    data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data', 'imgrec-potholes') # 'all_data'

    train_ids_labels = CSV_Helper()
    train_ids_labels._file_location = data_directory
    train_ids_labels._file_name = 'train_ids_labels'
    train_ids_labels._file_extension = '.csv'
    train_ids_labels._file_delimiter = ","
    train_ids_labels.get_dicts_from_csv()

    test_ids_only = CSV_Helper()
    test_ids_only._file_location = data_directory
    test_ids_only._file_name = 'test_ids_only'
    test_ids_only._file_extension = '.csv'
    test_ids_only._file_delimiter = ","
    test_ids_only.get_dicts_from_csv()

    images_directory = os.path.join(data_directory, 'all_data', 'all_data')

    # Training data. Rotate images and write to training_0 and training_1 directory
        # Create relevant directories
        # Read in images
        # Rotate
        # Write to relevant files
    train_dir = os.path.join(data_directory, 'train')
    shutil.rmtree(train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    train_dir_0 = os.path.join(train_dir, 'train_0')
    if not os.path.exists(train_dir_0):
        os.makedirs(train_dir_0)

    train_dir_1 = os.path.join(train_dir, 'train_1')
    if not os.path.exists(train_dir_1):
        os.makedirs(train_dir_1)

    print('--------------------TRAINING SET--------------------')
    ls_label_0 = []
    ls_label_1 = []
    training_images_len = len(train_ids_labels.data)
    for i, img_dict in enumerate(train_ids_labels.data):
        Image_ID = img_dict['Image_ID']
        Label = img_dict['Label']
        image_directory = os.path.join(images_directory, Image_ID)
        extension = '.JPG'
        image = Image_Helper()
        image.img_dir = image_directory
        image.read_ext = extension
        image.read_image()
        image.crop_image()
        image.rotate_image(angles=[90, 180, 270])
        if str(Label) == '0':
            image.new_dir = os.path.join(train_dir_0, Image_ID)
            ls_label_0.append(Image_ID)
        if str(Label) == '1':
            image.new_dir = os.path.join(train_dir_1, Image_ID)
            ls_label_1.append(Image_ID)
        image.reshape_image()
        image.reshape_rotated_images()
        print('[INFO] Writing image ' + str(i + 1) + ' out of ' + str(training_images_len) + ': ' + image.img_dir + ' to ' + image.new_dir)
        image.write_reshaped_image()
        image.write_reshaped_rotated_images()

    pickler.write_pickle(ls_label_0, os.path.join(data_directory, 'ls_label_0'))
    pickler.write_pickle(ls_label_1, os.path.join(data_directory, 'ls_label_1'))

    # Test (submission) data. Rotate images and write to submission file
    # Create relevant directories
    # Read in images
    # Rotate
    # Write to relevant files

    submission_dir = os.path.join(data_directory, 'submission')
    shutil.rmtree(submission_dir)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    print('--------------------SUBMISSION SET--------------------')
    ls_submission = []
    submission_images_len = len(test_ids_only.data)

    for i, img_dict in enumerate(test_ids_only.data):
        Image_ID = img_dict['Image_ID']
        ls_submission.append(Image_ID)
        image_directory = os.path.join(images_directory, Image_ID)
        extension = '.JPG'
        image = Image_Helper()
        image.img_dir = image_directory
        image.read_ext = extension
        image.read_image()
        image.crop_image()
        image.rotate_image(angles=[90, 180, 270])
        image.new_dir = os.path.join(submission_dir, Image_ID)
        image.reshape_image()
        image.reshape_rotated_images()
        print('[INFO] Writing image ' + str(i + 1) + ' out of ' + str(submission_images_len) + ': ' + image.img_dir + ' to ' + image.new_dir)
        image.write_reshaped_image()
        image.write_reshaped_rotated_images()

    pickler.write_pickle(ls_submission, os.path.join(data_directory, 'ls_submission'))