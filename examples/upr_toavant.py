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
    return X_train, X_test

def one_file(upr, file):
    depth = upr.read_depth(file)
    segments = []
    reward_function = []
    i = 0
    season = file.split("/")[1]
    telescope_integrated = 0
    work_done = 0
    a_A = 0.0020
    a_B = 0.0012
    alpha = (20/180)*3.142
    l = 1.35
    a = 0.0016
    if season == "autumn" or season == "winter":
        sensor = [35, 27, 28, 71, 72, 62, 15]
    else:
        sensor = [16, 17, 8, 1, 2]
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations = []
        # distance_travelled = upr.get_distance_travelled(file)
        for row in csv_reader:
            if (len(depth)>i):
                # distance_to_pile = distance_travelled[-1] - distance_travelled[i]
                P_A = float(row[sensor[1]])*100000
                P_B = float(row[sensor[2]])*100000
                F = a_A*P_A-a_B*P_B
                boom = float(row[sensor[3]])
                F_C = F * np.array([np.cos(boom)+alpha, np.sin(boom)+alpha])
                vx = float(row[sensor[5]])
                boom_dot = float(row[sensor[6]])
                v_C = np.array([vx-l*boom_dot*np.sin(boom)+a, l*boom_dot*np.cos(boom)+a])
                work_done += abs(np.dot(F_C, v_C))/20
                observation = [work_done,
                               boom, float(row[sensor[4]]), depth[i]]
                reward_i, segment = upr.get_intermediate_reward(observation)

                segments.append(segment)

                upr.combine_reward(reward_i, segment, i)
                reward = upr.reward
                reward_function.append(reward)
                demonstrations.append(observation)

                # print(' pred: ', segment)
                # print("intermediate reward: ", reward_i)
                # print('step: ', i)
                # print('reward: ', reward)
            i += 1

    demonstrations = np.array(demonstrations)
    segments = np.array(segments).reshape((len(segments), 1))
    data = np.hstack((demonstrations, segments))
    # upr.plot_data(data, "Testing", file)
    #
    # plt.plot(reward_function)
    # plt.ylabel('reward')
    # plt.xlabel('time')
    # plt.title(file)
    #
    # plt.show()

    reward_function = np.array(reward_function).reshape((len(reward_function), 1))
    data = np.hstack((data, reward_function))
    return data

def plot_all_data(all_data, files):
    row = 10
    if len(all_data)<10:
        row = len(all_data)
    col = 6
    j = 0
    plt.figure()
    for k in range(row):
        for i in range(col):
            place = j + i + 1
            plt.subplot(row, col, place)
            colour = "b-"
            if (i == 2 or i == 1):
                colour = "m-"
            if (i == 3):
                colour = "g-"
            if (i == 5):
                colour = "r-"

            plt.plot(all_data[k][:, i], colour)
            if j == 0:
                if (i == 0):
                    plt.title("Workdone")
                elif (i == 1):
                    plt.title("Boom")
                elif (i == 2):
                    plt.title("Bucket")
                elif (i == 3):
                    plt.title("Depth")
                elif (i == 4):
                    plt.title("Segment")
                elif (i == 5):
                    plt.title("Reward")

            if i == 0:
                plt.ylabel(files[k].split('/')[1])
        j += col
    plt.show()


def test(upr, files):
    all_data = []
    for file in files:
        upr.reset()
        data = one_file(upr, file)
        all_data.append(data)
    plot_all_data(all_data, files)

file_paths = get_file_paths(["data/winter"])
X_train, X_test = training_test_split(file_paths)
upr = UPR(X_train, n_clusters=3)
test(upr, X_test)

# upr = UPR(["data/autumn/19-17-30.csv"], n_clusters=3)
# file = "data/autumn/19-16-10.csv"
# one_file(upr, file)

# X_train = get_file_paths(["data/summer"])
# X_test = get_file_paths(["data/winter"])
# upr = UPR(X_train, n_clusters=3)
# test(upr, X_test)

# X_train = get_file_paths(["data/winter"])
# X_test = get_file_paths(["data/summer"])
# upr = UPR(X_train, n_clusters=3)
# test(upr, X_test)



