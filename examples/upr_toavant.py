import numpy as np
from irl.upr3 import UPR
import cv2
import matplotlib.pyplot as plt
import math
import csv
upr = UPR(["data/18-54-34.csv"])

reward_function = []

file = "data/19-15-30.csv"
segments = []
with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        observation = [i, abs(float(row[35]) - float(row[36])) / 100, float(row[27]),
                       abs(float(row[32]) - float(row[33])) / 100,
                       float(row[28])]
        reward_i, segment = upr.get_intermediate_reward(observation)
        print(' pred: ', segment)
        segments.append(segment)
        print("intermediate reward: ", reward_i)
        upr.combine_reward(reward_i, segment, i)
        reward = upr.reward
        print('step: ', i)
        print('reward: ', reward)
        reward_function.append(reward)
        i+=1

plt.plot(segments)
plt.ylabel("segment")
plt.xlabel('time')
plt.show()

plt.plot(reward_function)
plt.ylabel('reward')
plt.xlabel('time')
plt.show()


