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

def read_depth(file):
    time = file.split('/')[2].split('.csv')[0]
    folder = file.split('/')[0]+'/'+file.split('/')[1]+'_depth/'
    depth_file = folder+time + ".svo-depth.txt"
    f = open(depth_file, "r")
    i = 0
    depth = []
    for x in f:
        x = x.split()
        if x[0] != '#' and len(x) == 30 and i>35:
            diff = float(x[2])-float(x[20])
            depth.append(diff)
        i += 1

    return depth

def one_file(upr, file):
    depth = read_depth(file)
    segments = []
    reward_function = []
    i =0
    # with open(file) as csv_file:
    #     row_count = sum(1 for line in csv_file)
    #     print(row_count)
    # if (row_count > 200 and row_count < 400):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations =[]

        for row in csv_reader:
            if (len(depth)>i):
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
    plt.subplot(421)
    plt.title(file)
    plt.plot(demonstrations[:, 0], 'b')
    plt.ylabel('Transmission Pressure Difference ')
    plt.subplot(422)
    plt.plot(demonstrations[:, 1], 'r')
    plt.ylabel('Telescope Pressure')
    plt.subplot(423)
    plt.plot(demonstrations[:, 2], 'm')
    plt.ylabel('Boom Angle')
    plt.subplot(424)
    plt.plot(demonstrations[:, 3], 'm')
    plt.ylabel('Bucket angle')

    plt.subplot(427)
    plt.plot(segments)
    plt.ylabel("segment")
    plt.xlabel('time')

    plt.subplot(425)
    plt.plot(depth, 'g')
    plt.ylabel("depth / cm")

    plt.subplot(428)
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

# upr = UPR(["data/autumn/19-17-30.csv"], n_clusters=3)
# file = "data/autumn/19-16-10.csv"
# one_file(upr, file)

# X_train = get_file_paths(["data/autumn"])
# X_test = get_file_paths(["data/winter"])
# upr = UPR(X_train, n_clusters=3)
# test(upr, X_test)



