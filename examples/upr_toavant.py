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
    depth = upr.read_depth(file)
    segments = []
    reward_function = []
    i = 0
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations =[]
        distance_travelled = upr.get_distance_travelled(file)
        for row in csv_reader:
            if (len(depth)>i):
                distance_to_pile = distance_travelled[-1] - distance_travelled[i]
                observation = [abs(float(row[35])-float(row[36]))/100, float(row[27]),
                              float(row[71]), float(row[72]), depth[i]]#, float(row[62]), float(row[74])]
                # observation = [i, abs(float(row[35]) - float(row[36])) / 100, float(row[27])]
                reward_i, segment = upr.get_intermediate_reward(observation)

                print(' pred: ', segment)
                segments.append(segment)

                print("intermediate reward: ", reward_i)
                upr.combine_reward(reward_i, segment, i)
                reward = upr.reward
                print('step: ', i)
                print('reward: ', reward)
                reward_function.append(reward)
                demonstrations.append(observation)
            i += 1

    demonstrations = np.array(demonstrations)
    segments = np.array(segments).reshape((len(segments), 1))
    data = np.hstack((demonstrations, segments))
    upr.plot_data(data, "Testing", file)

    plt.plot(reward_function)
    plt.ylabel('reward')
    plt.xlabel('time')
    plt.title(file)

    plt.show()

def test(upr, files):
    for file in files:
        upr.reset()
        one_file(upr, file)


file_paths = get_file_paths(["data/autumn", "data/winter"])
X_train, X_test = training_test_split(file_paths)
upr = UPR(X_train, n_clusters=3)
test(upr, X_test)

# upr = UPR(["data/autumn/19-17-30.csv"], n_clusters=3)
# file = "data/autumn/19-16-10.csv"
# one_file(upr, file)

# X_train = get_file_paths(["data/autumn"])
# X_test = get_file_paths(["data/winter"])
# upr = UPR(X_train, n_clusters=3)
# test(upr, X_test)



