import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import datetime

from data_and_models.initial_data_load import initial_data_load
from data_and_models.model_builder import create_models
from Helpers.PickleHelper import Pickle_Helper
from Helpers.ModelHelpers import ImageModelHelper


INITIAL_DATA_LOAD = False
TRAIN_NEW_MODELS = True
SUBMISSION = True
NUM_OF_MODELS = 50
PER_GROUP_SUBSET_SIZE = 1000

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data', 'imgrec-potholes')
pickler = Pickle_Helper()
print(str(datetime.datetime.now()) + ' [INFO] STARTING PROCESS')
if INITIAL_DATA_LOAD:
    print(str(datetime.datetime.now()) + ' [INFO] INITIALIZING DATA LOAD')
    initial_data_load()
    print(str(datetime.datetime.now()) + ' [INFO] DATA LOAD COMPLETED')

if TRAIN_NEW_MODELS:
    print(str(datetime.datetime.now()) + ' [INFO] TRAINING MODELS')
    for i in range(1, NUM_OF_MODELS+1):
        print(str(datetime.datetime.now()) + ' [INFO] CREATING MODEL ' + str(i) + '')
        create_models('./model_objects/model_' + str(i), PER_GROUP_SUBSET_SIZE)
    print(str(datetime.datetime.now()) + ' [INFO] MODEL TRAINING COMPLETED')

if SUBMISSION:
    print(str(datetime.datetime.now()) + ' [INFO] STARTING SUBMISSION')
    ls_submission = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_submission.dat'))
    sub_len = len(ls_submission)
    return_submission = []
    image_model = ImageModelHelper()
    image_model.data_directory = data_directory
    model_objects = []
    for i in range(1, NUM_OF_MODELS+1):
        model_objects.append(pickler.read_pickle('./model_objects/model_' + str(i) + '.dat'))
    for count, sub_img in enumerate(ls_submission):
        print(str(datetime.datetime.now()) + ' [INFO] IMAGE ' + str(count+1) + ' OUT OF ' + str(sub_len))
        pred_df = pd.DataFrame()
        image_model.tvt_labels['submission'] = [sub_img]
        image_model.select_rotations(3, 'submission')
        image_model.build_feature_labels('submission')
        for mod in model_objects:
            pred_df = pred_df.append(
                image_model.predict(model=mod.model,
                                    X=image_model.Xy_submission['X'],
                                    y=None,
                                    labels=image_model.tvt_labels['submission'],
                                    prob_cutoff=0.5),
                ignore_index=True)
        prediction = pred_df.apply(np.median, axis=0)[1]
        if prediction > 0.5:
            prediction = pred_df.apply(np.max, axis=0)[1]
        else:
            prediction = pred_df.apply(np.min, axis=0)[1]
        data = OrderedDict()
        data['Image_ID'] = sub_img
        data['Label'] = prediction
        return_submission.append(data)
    return_submission_df = pd.DataFrame(return_submission)
    return_submission_df.to_csv('JurgSubmission.csv', index=False)

print(str(datetime.datetime.now()) + ' [INFO] PROCESS COMPLETED')
