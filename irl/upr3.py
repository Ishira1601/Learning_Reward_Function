import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import math
class UPR:
    def __init__(self, files, n_clusters):
        self.files = files
        self.reward = 0
        self.demonstrations = []
        self.expert = []
        self.y = []
        self.X = []
        self.load_data()
        self.the_stages = []
        self.n_clusters = n_clusters
        self.stages()
        self.step_classifier()
        self.T = 0

    def load_data(self):
        d = 0
        k = 0
        for file in self.files:
            i = 0
            depth = self.read_depth(file)
            # with open(file) as csv_file:
            #     row_count = sum(1 for line in csv_file)
            # if (row_count > 200 and row_count < 400):
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if (len(depth) > i):
                        observation = [k, abs(float(row[35])-float(row[36]))/100, float(row[27]),
                                      float(row[71]), float(row[72]), depth[i]] #, float(row[62]), float(row[74])]
                        # observation = [k, i, abs(float(row[35]) - float(row[36])) / 100, float(row[27])]
                        d = len(observation)
                        self.demonstrations.append(observation)
                        i += 1
            if k==0:
                self.T = i
            k+=1

        self.demonstrations = np.array(self.demonstrations)
        self.expert = self.demonstrations[:, 1:d]
        plt.subplot(321)
        plt.plot(self.expert[:, 0], 'b')
        plt.ylabel('Transmission Pressure Difference ')
        plt.subplot(322)
        plt.plot(self.expert[:, 1], 'r')
        plt.ylabel('Telescope Pressure')
        plt.subplot(323)
        plt.plot(self.expert[:, 2], 'm')
        plt.ylabel('Boom Angle')
        plt.subplot(324)
        plt.plot(self.expert[:, 3], 'm')
        plt.ylabel('Bucket angle')
        plt.subplot(325)
        plt.plot(self.expert[:, 4], 'g')
        plt.ylabel('depth')
        # plt.subplot(326)
        # plt.plot(self.expert[:, 5], 'm')
        # plt.ylabel('angle position')
        plt.xlabel('time')
        plt.show()

    def read_depth(self, file):
        time = file.split('/')[2].split('.csv')[0]
        folder = file.split('/')[0] + '/' + file.split('/')[1] + '_depth/'
        depth_file = folder + time + ".svo-depth.txt"
        f = open(depth_file, "r")
        i = 0
        depth = []
        for x in f:
            x = x.split()
            if x[0] != '#' and len(x) == 30 and i>35:
                diff = float(x[2]) - float(x[20])
                depth.append(diff)
            i += 1

        return depth

    def stages(self):
        self.X = self.expert
        cluster_centers = self.set_cluster_centers()
        cluster_centers = np.array(cluster_centers)
        clusters = KMeans(n_clusters=self.n_clusters, init=cluster_centers).fit(self.X)
        self.y = clusters.labels_
        plt.plot(self.y)
        plt.show()
        self.to_stages(self.y)

    def set_cluster_centers(self):
        i = round(self.T / (2 * self.n_clusters))
        cluster_centers = []
        k = 1
        j = k * i
        while j<self.T:
            cluster_centers.append(self.expert[j])
            k += 2
            j = k * i
        return cluster_centers


    def to_stages(self, y):
        n= len(y)
        stages = []
        samples = self.expert
        for i in range(n):
            if stages==[]:
                stages.append([samples[i]])
            elif len(stages)>y[i]:
                stages[y[i]].append(samples[i])
            else:
                stages.append([samples[i]])

        for stage in stages:
            mu_and_sigma = self.get_mean_and_variance(np.array(stage))
            self.the_stages.append(mu_and_sigma)
        self.the_stages = np.array(self.the_stages)

    def get_mean_and_variance(self, x):
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        return np.array([mu_x, sigma_x])

    def reset(self):
        self.reward = 0

    def step_classifier(self):
        self.clf = KNeighborsClassifier()
        self.clf.fit(self.X, self.y)

    def get_intermediate_reward(self, state):
        n = len(state)-1
        segment = self.clf.predict([state])[0]
        expert_t = self.the_stages[segment]
        if (segment+1<self.n_clusters):
            expert_t = self.the_stages[segment+1]
        mu_t = expert_t[0]
        sigma_t = expert_t[1]
        summed = 0
        for j in range(n):
            dist = (np.square(state[j] - mu_t[j])) / np.square(sigma_t[j])
            if not math.isnan(dist) and not math.isinf(dist):
                summed = summed + dist
            else:
                continue
        # reward_t = n/summed
        reward_t = 100 - summed
        return reward_t, segment

    def combine_reward(self, reward_i, segment, time):
        # if (time>300):
        #     self.reward -= time*100

        if segment>0:
            # self.reward += reward_i*pow(2, segment-1)
            self.reward = reward_i * pow(2, segment - 1)
        else:
            self.reward = 0





