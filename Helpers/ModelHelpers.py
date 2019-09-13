import math
import random
import os
from abc import ABC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, log_loss

from Helpers.PickleHelper import Pickle_Helper


class ModelHelper(ABC):

    def __init__(self):
        super().__init__()
        self.labels = None
        self.train_perc = None
        self.validate_perc = None
        self.test_perc = None
        self.pickler = Pickle_Helper()
        self.or_tvt_labels = {'train': [],
                              'validate': [],
                              'test': [],
                              'submission': []}
        self.tvt_labels = {'train': [],
                           'validate': [],
                           'test': [],
                           'submission': []}

    def train_validate_test(self):
        list_len = len(self.labels)
        train_size = math.floor(list_len * self.train_perc)
        validate_size = math.floor(list_len * self.validate_perc)
        test_size = math.floor(list_len * self.test_perc)
        list_index = [i for i in range(0, list_len)]
        random.shuffle(list_index)
        train = [self.labels[i] for i in list_index[0:train_size]]
        validate = [self.labels[i] for i in list_index[train_size:train_size + validate_size]]
        test = [self.labels[i] for i in list_index[train_size + validate_size:]]
        self.tvt_labels['train'] = train
        self.tvt_labels['validate'] = validate
        self.tvt_labels['test'] = test
        self.or_tvt_labels = self.tvt_labels


class ImageModelHelper(ModelHelper):

    def __init__(self):
        ModelHelper.__init__(self)
        self.data_directory = None
        self.Xy_train = None
        self.Xy_validate = None
        self.Xy_test = None
        self.Xy_submission = None
        self.models = []
        self.final_model = None
        self.pred_df = None
        self.gof_list = []

    def select_rotations(self, rot_amount, to):
        ls_with_rotations = []
        for img in self.tvt_labels[to]:
            for i in range(1, rot_amount + 1):
                ls_with_rotations.append(img + '_' + str(i))
        self.tvt_labels[to].extend(ls_with_rotations)

    def build_feature_labels(self, to):
        X = []
        y = []
        if not to == 'submission':
            for file in self.tvt_labels[to]:
                file_name = os.path.join(self.data_directory, 'train', 'train_0', file + '.dat')
                if os.path.isfile(file_name):
                    # shutil.copyfile(file_name, train_data_dir)
                    pickle_in = self.pickler.read_pickle(file_name)
                    X.append(pickle_in)
                    y.append(0)
                file_name = os.path.join(self.data_directory, 'train', 'train_1', file + '.dat')
                if os.path.isfile(file_name):
                    # shutil.copyfile(file_name, train_data_dir)
                    pickle_in = self.pickler.read_pickle(file_name)
                    X.append(pickle_in)
                    y.append(1)
        else:
            for file in self.tvt_labels[to]:
                file_name = os.path.join(self.data_directory, 'submission', file + '.dat')
                if os.path.isfile(file_name):
                    # shutil.copyfile(file_name, train_data_dir)
                    pickle_in = self.pickler.read_pickle(file_name)
                    X.append(pickle_in)
                    y.append(999)

        X = np.array(X).reshape(len(X), -1)
        if to == 'train':
            self.Xy_train = {
                'X': X,
                'y': y
            }
        elif to == 'validate':
            self.Xy_validate = {
                'X': X,
                'y': y
            }
        elif to == 'test':
            self.Xy_test = {
                'X': X,
                'y': y
            }
        elif to == 'submission':
            self.Xy_submission = {
                'X': X,
                'y': y
            }

        return X, y

    def random_Forest(self, min_sample_split=None):
        if min_sample_split:
            min_samples_splits = [min_sample_split]
        else:
            min_samples_splits = [0.05, 0.1, 0.2, 0.5]
        mod_log_loss = 99999999
        final_mod = None
        for min_split in min_samples_splits:
            clf = RandomForestClassifier(n_estimators=40, min_samples_split=min_split)
            clf.fit(X=self.Xy_train['X'], y=self.Xy_train['y'])
            challenge_probs = clf.predict_proba(self.Xy_validate['X'])
            challenge_log_loss = log_loss(self.Xy_validate['y'], challenge_probs)
            if challenge_log_loss < mod_log_loss:
                mod_log_loss = challenge_log_loss
                final_mod = clf

        self.models.append(final_mod)

    def gradient_boost(self, learn_rate=None):
        if learn_rate:
            lr_list = [learn_rate]
        else:
            lr_list = [0.05, 0.1, 0.2, 0.3]
        mod_log_loss = 99999999
        final_mod = None
        for lr in lr_list:
            clf = XGBClassifier(n_estimators=5, learning_rate=lr)
            clf.fit(X=self.Xy_train['X'], y=self.Xy_train['y'])
            challenge_probs = clf.predict_proba(self.Xy_validate['X'])
            challenge_log_loss = log_loss(self.Xy_validate['y'], challenge_probs)
            if challenge_log_loss < mod_log_loss:
                mod_log_loss = challenge_log_loss
                final_mod = clf

        self.models.append(final_mod)

    def ada_boost(self, learn_rate=None):
        if learn_rate:
            lr_list = [learn_rate]
        else:
            lr_list = [0.05, 0.1, 0.2, 0.3]
        mod_log_loss = 99999999
        final_mod = None
        for lr in lr_list:
            clf = AdaBoostClassifier(n_estimators=10, learning_rate=lr)
            clf.fit(X=self.Xy_train['X'], y=self.Xy_train['y'])
            challenge_probs = clf.predict_proba(self.Xy_validate['X'])
            challenge_log_loss = log_loss(self.Xy_validate['y'], challenge_probs)
            if challenge_log_loss < mod_log_loss:
                mod_log_loss = challenge_log_loss
                final_mod = clf

        self.models.append(final_mod)

    def model_select(self):
        mod_log_loss = 99999999
        for model in self.models:
            challenge_probs = model.predict_proba(self.Xy_test['X'])
            challenge_log_loss = log_loss(self.Xy_test['y'], challenge_probs)
            if challenge_log_loss < mod_log_loss:
                mod_log_loss = challenge_log_loss
                self.final_model = model

    def predict(self, model, X, y, labels, prob_cutoff):
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

    def gof_stats(self, y, y_pred):
        self.gof_list.append(classification_report(y, y_pred))


class FinalImageModel:

    def __init__(self):
        self.pickler = Pickle_Helper()
        self.model = None
        self.gof_list = None
        self.or_tvt_labels = None



