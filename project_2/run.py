# project_2.py

import numpy as np

from matplotlib import pyplot as plt

import argparse
import csv

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
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
    n = Y.shape[1]
    error = 1/(2*n) * np.sum((Y_true(:, 1) == Y).all()) + 1/(2*n)

def run():
    X_read = read_file("data/train.csv")
    Y_read = read_file("data/train_y.csv")

    X_data = X_read(9:,:)
    Y_data = Y_read

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
    prediction = classifier.predict(X_test)



    try:
        if ipython:
            ip.embed()
    except:
        print("Well, you're supposed to have iPython but something went wrong...")

if __name__ == '__main__':
    run()
