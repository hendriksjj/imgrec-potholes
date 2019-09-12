from abc import ABC, abstractmethod
import csv
from numpy import genfromtxt

class CSV_Helper(ABC):

    def __init__(self):
        super().__init__()
        self._file_location = None
        self._file_name = None
        self._file_extension = None
        self._file_delimiter = ","
        self.data = None
        self.id = []

    def get_dicts_from_csv(self):
        try:
            with open(self._file_location + "\\" + self._file_name + '.csv', mode='r', newline='\n') as infile:
                reader = csv.DictReader(infile, delimiter=self._file_delimiter, quotechar='"')
                result = list(reader)
            self.data = result
        except Exception as error:
            print("CSV Load Error: ", error, "\n--WITH File--\n", self._file_name)

    def get_lists_from_csv(self):
        # try:
        with open(self._file_location + "\\" + self._file_name + '.csv', mode='r', newline='\n') as infile:
            data = list(csv.reader(infile))
        self.data = data

