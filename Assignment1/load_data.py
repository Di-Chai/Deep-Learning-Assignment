import numpy as np
from local_path import data_path
import csv


def loadCSVfile(fileName, Dtype):
    list_file = []
    with open(data_path + fileName, 'r') as csv_file:
        all_lines = csv.reader(csv_file)
        for one_line in all_lines:
            if Dtype is 'float':
                list_file.append([float(e) for e in one_line])
            if Dtype is 'int':
                list_file.append([int(float(e)) for e in one_line])
    csv_file.close()
    data = np.array(list_file)
    return data