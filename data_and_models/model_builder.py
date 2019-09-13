import os
import random

from data_and_models.initial_data_load import create_data_directories
from Helpers.PickleHelper import Pickle_Helper
from Helpers.ModelHelpers import ImageModelHelper
from Helpers.ModelHelpers import FinalImageModel


def create_models(model_name, per_group_subset_size):

    data_directory, model_data_dir, train_data_dir, validate_data_dir, test_data_dir, submission_data_dir = create_data_directories()

    # Loading in the training labels
    pickler = Pickle_Helper()

    ls_label_0 = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_label_0.dat'))
    ls_label_1 = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_label_1.dat'))


    # Take random subsets from ls_label_0, ls_label_1. Can take full set if you have enough RAM

    ls_select = []

    ls_label_0_select = random.sample(ls_label_0, k=per_group_subset_size)
    ls_label_1_select = random.sample(ls_label_1, k=per_group_subset_size)

    ls_select.extend(ls_label_0_select)
    ls_select.extend(ls_label_1_select)

    # Initialize tvt
    image_model = ImageModelHelper()
    image_model.data_directory = data_directory
    image_model.labels = ls_select
    # Train, Validate, Test, submission
    image_model.train_perc = 0.7
    image_model.validate_perc = 0.2
    image_model.test_perc = 0.1
    image_model.train_validate_test()
    # image_model.tvt_labels['submission'] = ls_submission

    # Add rotations
    image_model.select_rotations(3, 'train')
    image_model.select_rotations(3, 'validate')
    image_model.select_rotations(3, 'test')
    # image_model.select_rotations(3, 'submission')

    # Build features and labels
    image_model.build_feature_labels('train')
    image_model.build_feature_labels('validate')
    image_model.build_feature_labels('test')
    # image_model.build_feature_labels('submission')

    # Fit random forest
    # image_model.random_Forest()
    # image_model.gradient_boost()
    image_model.ada_boost()
    image_model.model_select()

    # Predict testing set
    rf_pred_df = image_model.predict(image_model.final_model,
                                     image_model.Xy_test['X'],
                                     image_model.Xy_test['y'],
                                     image_model.tvt_labels['test'],
                                     0.5)

    image_model.gof_stats(rf_pred_df['y'], rf_pred_df['y_pred'])

    final_image_model = FinalImageModel()
    final_image_model.model = image_model.final_model
    final_image_model.gof_list = image_model.gof_list
    final_image_model.or_tvt_labels = image_model.or_tvt_labels

    final_image_model.pickler.write_pickle(final_image_model, model_name)

