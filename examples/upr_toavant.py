import numpy as np
from irl.upr3 import UPR
import cv2
import matplotlib.pyplot as plt
import math
import csv

def get_file_paths(folders):
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

def training_test_split(X):
    from random import random
    from random import seed
    X_train = []
    X_test = []
    seed(16)
    for x in X:
        r = random()
        if r<0.8:
            X_train.append(x)
        else:
            X_test.append(x)
    return  X_train, X_test

def one_file(upr, file):
    segments = []
    reward_function = []
    i =0
    with open(file) as csv_file:
        row_count = sum(1 for line in csv_file)
        print(row_count)
    if (row_count > 200 and row_count < 400):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                observation = [i, abs(float(row[35]) - float(row[36])) / 100, float(row[27])]
                # ,abs(float(row[32])-float(row[33]))/100,
                # float(row[28])]
                reward_i, segment = upr.get_intermediate_reward(observation)
                print(' pred: ', segment)
                segments.append(segment)
                print("intermediate reward: ", reward_i)
                upr.combine_reward(reward_i, segment, i)
                reward = upr.reward
                print('step: ', i)
                print('reward: ', reward)
                reward_function.append(reward)
                i += 1
        plt.subplot(211)
        plt.plot(segments)
        plt.ylabel("segment")
        plt.xlabel('time')
        plt.title(file)

        plt.subplot(212)
        plt.plot(reward_function)
        plt.ylabel('reward')
        plt.xlabel('time')
        plt.show()

def test(upr, files):
    for file in files:
        upr.reset()
        one_file(upr, file)

file_paths = get_file_paths(["data/autumn", "data/winter"])
X_train, X_test = training_test_split(file_paths)
upr = UPR(X_train, n_clusters=3)
test(upr, X_test)
# upr = UPR(["data/autumn/19-01-52.csv", "data/autumn/19-15-30.csv", "data/winter/14-31-37.csv", "data/winter/14-32-16.csv"], n_clusters=8)
# file = "data/autumn/19-34-32.csv"
# one_file(upr, file)




