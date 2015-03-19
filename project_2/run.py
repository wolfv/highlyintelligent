# project_2.py

import numpy as np

from matplotlib import pyplot as plt

import argparse
import csv

ipython = False
try:
	import IPython as ip
	ipython = True
except:
	print("No IPython, wie schade ... :(")

parser = argparse.ArgumentParser()
parser.add_argument("--testfile", "-t", help="Select XML file")


def read_file(filename):
	result = []
	with open(filename) as train_fp:
		reader = csv.reader(train_fp)
		for row in reader:
			result.append([int(el) for el in row])

	return np.array(result)

def run():
	X_read = read_file("data/train.csv")
	Y_read = read_file("data/train_y.csv")
	try:
		if ipython:
			ip.embed()
	except:
		print("Well, you're supposed to have iPython but something went wrong...")

if __name__ == '__main__':
	run()
