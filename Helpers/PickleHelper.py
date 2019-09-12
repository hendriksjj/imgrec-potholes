import os
import pickle

class Pickle_Helper:

    def __init__(self):
        pass

    @staticmethod
    def write_pickle(res, file_name):
        # fileName = fileName.replace("\\","_")
        file = open(file_name + ".dat", "wb")
        pickle.dump(res, file)
        file.close()

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, 'rb') as f:
            pic = pickle.load(f)
        return pic