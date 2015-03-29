# project_2.py

import numpy as np

from matplotlib import pyplot as plt

import argparse
import csv

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

import neurolab as nl


ipython = False
try:
    import IPython as ip
    ipython = True
except:
    print("No IPython, wie schade ... :(")

parser = argparse.ArgumentParser()
parser.add_argument("--testfile", "-t", help="Select XML file")
parser.add_argument("--samplesize", "-s", help="Choose how many runs", default=1)
parser.add_argument("--interactive", "-i", help="Interactive?", default=False)


def read_file(filename):
    # This reads the file and outputs a numpy array, 
    # filled with integers 
    result = []
    with open(filename) as train_fp:
        reader = csv.reader(train_fp)
        for row in reader:
            result.append([int(el) for el in row])

    return np.array(result)

def error(Y_true, Y):
    # calculate error as described in the pdf
    n = Y.shape[0]
    error = 1/(2*n) * np.sum(Y_true[:, 0] != Y[:, 0]) + 1/(2*n) * np.sum(Y_true[:, 1] != Y[:, 1])
    return error

def binarize(val, steps, append_to=None):
    val_min =  np.min(val)
    # stepsize = (np.max(val) - val_min) / (steps + 1)
    
    val_steps = (val - val_min) / (np.max(val) - val_min + 0.00000001) * (steps)
    ret = np.zeros((val.shape[0], steps ))
    val = np.int_(np.floor(val_steps))


    for i in range(0, val.shape[0]):
        ret[i, val[i]] = 1

    if append_to is not None:
        ret_arr = np.concatenate((append_to, ret), axis=1)
        return ret_arr
    return ret

# def inverse_transform(res_arr):
#     for line in res_arr:



def run(args):
    X_read = read_file("data/train.csv")

    Y_read = read_file("data/train_y.csv")

    X_data = X_read
    Y_data = Y_read


    # X_data = binarize(X_read[:, 0], 5, append_to=X_data)
    # X_data = binarize(X_read[:, 1], 5, append_to=X_data)
    # X_data = binarize(X_read[:, 2], 5, append_to=X_data)
    # X_data = binarize(X_read[:, 3], 5, append_to=X_data)
    # X_data = binarize(X_read[:, 4], 10, append_to=X_data)
    # X_data = binarize(X_read[:, 5], 10, append_to=X_data)
    # X_data = binarize(X_read[:, 6], 5, append_to=X_data)
    # X_data = binarize(X_read[:, 7], 10, append_to=X_data)
    # X_data = binarize(X_read[:, 8], 10, append_to=X_data)
    # X_data = binarize(X_read[:, 9], 5, append_to=X_data)


    for i in range(0, int(args.samplesize)):
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)
        
        input_type = np.zeros((X_data.shape[1], 2))
        input_type[:, 1] = 1
        for j in range(0, 9):
            input_type[j, :] = np.array([np.min(X_data[:,j]), np.max(X_data[:,j])])

        network = nl.net.newff(input_type, [X_data.shape[1], 10])
        network.trainf = nl.train.train_bfgs
        # for n in network.layers:
        #     print(n.ci)
        bin_input = binarize(Y_data[:, 0], 3)
        bin_input = binarize(Y_data[:, 1], 7, bin_input)
        network.train(X_data, bin_input, show=1)

        bin_input = binarize(Y_test[:, 0], 3)
        bin_input = binarize(Y_test[:, 1], 7, bin_input)
        result = network.sim(X_test)
        print(result)
        print(Y_test)
        # result = inverse_transform(result)

        # print("%s Run, Error: %s" % (i, error(Y_test, result)))

    if args.testfile:
        X_testfile = read_file(args.testfile)
        Y_testfile_prediction = np.zeros((X_testfile.shape[0], 2), dtype=np.int)
        X_data = X_testfile[:,9:]
        Y_testfile_prediction[:, 0] = classifier_y0.predict(X_data)
        # Y_testfile_prediction[:, 1] = classifier_y1.predict(X_data)
        np.savetxt('result.txt', Y_testfile_prediction,  fmt='%i', delimiter=",")

    try:
        if ipython and args.interactive:
            ip.embed()
    except:
        print("Well, you're supposed to have iPython but something went wrong...")

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
