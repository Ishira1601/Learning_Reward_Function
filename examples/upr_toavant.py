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
    work_done = 0
    workdone_x = 0
    workdone_y = 0
    a_A = 0.0020
    a_B = 0.0012
    alpha = (20/180)*3.142
    a = 0.0016
    prev_boom = 0
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations = []
        # distance_travelled = upr.get_distance_travelled(file)
        for row in csv_reader:
            if (len(depth)>i):
                if season == "autumn" or season == "winter":
                    P_A = float(row[28])*100000
                    P_B = float(row[27])*100000
                    boom = float(row[71])
                    bucket = float(row[72])
                    vx = float(row[62])
                    l = float(row[21])
                if season == "summer":
                    P_A = float(row[9]) * 100000
                    P_B = float(row[8]) * 100000
                    boom = float(row[1])
                    bucket = float(row[2])
                    vx = 0.0
                    l = float(row[3])
                F = a_A*P_A-a_B*P_B
                F_C = F * np.array([np.cos(boom), np.sin(boom)])
                boom_dot = (boom - prev_boom) * 15
                prev_boom = boom
                v_C = np.array([vx-l*boom_dot*np.sin(boom)+a, l*boom_dot*np.cos(boom)+a])
                work_done += abs(np.dot(F_C, v_C))/15
                workdone_x += abs(F_C[0] * v_C[0])/15
                workdone_y += abs(F_C[1] * v_C[1])/15
                F_Re = np.linalg.norm(F_C)
                v_Re = np.linalg.norm(v_C)
                observation = [F, P_A, P_B, F_C[0], F_C[1], v_C[0], v_C[1],
                           boom, bucket, l]
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
    col = all_data[0].shape[1]
    j = 0
    plt.figure()
    for k in range(row):
        for i in range(col):
            place = j + i + 1
            plt.subplot(row, col, place)
            colour = "b-"
            if (i == 3 or i == 4 or i == 7 or i == 8):
                colour = "m-"
            if (i == 5 or i == 6 or i == 9):
                colour = "g-"
            if (i == 11):
                colour = "r-"

            plt.plot(all_data[k][:, i], colour)
            if j == 0:
                if (i == 0):
                    plt.title("aAPA - aBPB")
                elif (i == 1):
                    plt.title("P_A")
                elif (i == 2):
                    plt.title("P_B")
                elif (i == 3):
                    plt.title("Force-x")
                elif (i == 4):
                    plt.title("Force-y")
                elif (i == 5):
                    plt.title("Vel-x")
                elif (i == 6):
                    plt.title("Vel-y")
                elif (i == 7):
                    plt.title("Boom")
                elif (i == 8):
                    plt.title("Bucket")
                elif (i == 9):
                    plt.title("Length")

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

file_paths = get_file_paths(["data/winter", "data/summer"])
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



