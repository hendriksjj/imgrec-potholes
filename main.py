import os
import random
import shutil
import numpy as np
import pandas as pd
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from initialization.initial_data_load import initial_load
from Helpers.PickleHelper import Pickle_Helper

if False:
    initial_load()

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

all_obs = os.listdir(os.path.join(data_directory, 'train', 'train_0'))
all_obs.extend(os.listdir(os.path.join(data_directory, 'train', 'train_1')))
# Loading in the training labels
pickler = Pickle_Helper()

ls_label_0 = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_label_0.dat'))
ls_label_1 = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_label_1.dat'))
ls_submission = pickler.read_pickle(file_name=os.path.join(data_directory, 'ls_submission.dat'))

# Take random subsets from ls_label_0, ls_label_1. Can take full set if you have enough RAM

ls_label_0_select = random.sample(ls_label_0, k=200)
ls_label_1_select = random.sample(ls_label_1, k=200)

ls_select = []
ls_select.extend(ls_label_0_select)
ls_select.extend(ls_label_1_select)
# Train, Validate, Test
def train_validate_test(ls, train_size, validate_size, test_size):
    list_len = len(ls)
    train_size = math.floor(list_len * train_size)
    validate_size = math.floor(list_len * validate_size)
    test_size = math.floor(list_len * test_size)
    list_index = [i for i in range(0, list_len)]
    random.shuffle(list_index)
    train = [ls[i] for i in list_index[0:train_size]]
    validate = [ls[i] for i in list_index[train_size:train_size + validate_size]]
    test = [ls[i] for i in list_index[train_size + validate_size:]]
    return train, validate, test

def select_rotations(ls, rot_amount):
    ls_with_rotations = []
    for img in ls:
        ls_with_rotations.append(img)
        for i in range(1, rot_amount + 1):
            ls_with_rotations.append(img + '_' + str(i))
    return ls_with_rotations

ls_train, ls_validate, ls_test = train_validate_test(ls_select, 0.7, 0.2, 0.1)

# Add rotations
ls_train_full = select_rotations(ls_train, 3)
ls_validate_full = select_rotations(ls_validate, 3)
ls_test_full = select_rotations(ls_test, 3)
ls_submission_full = select_rotations(ls_test, 3)

# Copy files over to model_train folder
# Create train, validate, test arrays

def build_feature_labels(data_directory, label_list, submission=False):
    X = []
    y = []
    if not submission:
        for file in label_list:
            file_name = os.path.join(data_directory, 'train', 'train_0', file + '.dat')
            if os.path.isfile(file_name):
                # shutil.copyfile(file_name, train_data_dir)
                pickle_in = pickler.read_pickle(file_name)
                X.append(pickle_in)
                y.append(0)
            file_name = os.path.join(data_directory, 'train', 'train_1', file + '.dat')
            if os.path.isfile(file_name):
                # shutil.copyfile(file_name, train_data_dir)
                pickle_in = pickler.read_pickle(file_name)
                X.append(pickle_in)
                y.append(1)
    else:
        for file in label_list:
            file_name = os.path.join(data_directory, 'submission', file + '.dat')
            if os.path.isfile(file_name):
                # shutil.copyfile(file_name, train_data_dir)
                pickle_in = pickler.read_pickle(file_name)
                X.append(pickle_in)
                y.append(999)

    X = np.array(X).reshape(len(X), -1)
    # y = np.array(y).reshape(len(y), -1)
    return X, y

X_train, y_train = build_feature_labels(data_directory, ls_train_full)
X_validate, y_validate = build_feature_labels(data_directory, ls_validate_full)
X_test, y_test = build_feature_labels(data_directory, ls_test_full)

pass

clf = RandomForestClassifier()
print(clf)

clf.fit(X_train, y_train)

def predict(model, X, y, labels, prob_cutoff):
    preds = model.predict_proba(X)

    labels_non_unique = [img.split('_', 1)[0] for img in labels]
    df = pd.DataFrame(preds)
    del df[0]
    if y:
        df['y'] = y

    df = df.set_index(pd.Index(labels_non_unique))

    df_group_by = df.groupby(df.index)
    predict_probs = df_group_by.mean()
    predict_probs['y_pred'] = predict_probs[1] > prob_cutoff
    predict_probs[['y_pred']] = predict_probs[['y_pred']].astype(int)
    return predict_probs

test_pred = predict(clf, X_test, y_test, ls_test_full, prob_cutoff=0.5)
#
clf_report = classification_report(test_pred['y'], test_pred['y_pred'])
print(clf_report)
