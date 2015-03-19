# project_2.py

import numpy as np

from matplotlib import pyplot as plt

import argparse
import csv

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

ipython = False
try:
    import IPython as ip
    ipython = True
except:
    print("No IPython, wie schade ... :(")

parser = argparse.ArgumentParser()
parser.add_argument("--testfile", "-t", help="Select XML file")
parser.add_argument("--samplesize", "-s", help="Select XML file", default=1)


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


def run(args):
    X_read = read_file("data/train.csv")
    Y_read = read_file("data/train_y.csv")

    X_data = X_read[:,9:]
    Y_data = Y_read
    X_data = binarize(X_read[:, 0], 5, append_to=X_data)
    X_data = binarize(X_read[:, 1], 5, append_to=X_data)
    X_data = binarize(X_read[:, 2], 5, append_to=X_data)
    X_data = binarize(X_read[:, 3], 5, append_to=X_data)
    X_data = binarize(X_read[:, 4], 5, append_to=X_data)
    X_data = binarize(X_read[:, 5], 5, append_to=X_data)
    X_data = binarize(X_read[:, 6], 5, append_to=X_data)
    X_data = binarize(X_read[:, 7], 5, append_to=X_data)
    X_data = binarize(X_read[:, 8], 5, append_to=X_data)
    X_data = binarize(X_read[:, 9], 5, append_to=X_data)
    
    for i in range(0, int(args.samplesize)):
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

        classifier_y0 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train[:,0])
        prediction = np.zeros(Y_test.shape)
        prediction[:,0] = classifier_y0.predict(X_test)
        
        classifier_y1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train[:,1])
        prediction[:,1] = classifier_y1.predict(X_test)

        print("%s Run, Error: %s" % (i, error(Y_test, prediction)))

    if args.testfile:
        X_testfile = read_file(args.testfile)
        Y_testfile_prediction = np.zeros((X_testfile.shape[0], 2), dtype=np.int)
        X_data = X_testfile[:,9:]
        Y_testfile_prediction[:, 0] = classifier_y0.predict(X_data)
        Y_testfile_prediction[:, 1] = classifier_y1.predict(X_data)
        np.savetxt('result.txt', Y_testfile_prediction,  fmt='%i', delimiter=",")

    try:
        if ipython:
            ip.embed()
    except:
        print("Well, you're supposed to have iPython but something went wrong...")

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
