import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
import warnings
warnings.filterwarnings("ignore")
import math
class UPR:
    def __init__(self, files):
        self.files = files
        self.observations = []
        self.segments = []
        self.expert_segments = []
        self.reward = 0
        self.demonstrations = []
        self.expert = []
        self.y = []
        self.X = []
        self.get_mean_std_expert()
        # self.to_segment()
        self.the_stages = []
        self.n_clusters = 3
        self.stages(self.n_clusters)
        # self.clustering(3)
        self.step_classifier()


    def get_mean_std_expert(self):
        d = 0
        k = 0
        for file in self.files:
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                i = 0
                for row in csv_reader:
                    observation = [k, i, abs(float(row[35])-float(row[36])), float(row[27]),
                                   abs(float(row[32])-float(row[33])),
                                   float(row[28])]
                    d = len(observation)
                    self.demonstrations.append(observation)
                    i += 1
            k+=1

        self.demonstrations = np.array(self.demonstrations)
        self.expert = self.demonstrations[:, 1:d]
        plt.subplot(211)
        plt.plot(self.expert[:, 1], 'b')
        plt.plot(self.expert[:, 3], 'c')
        plt.ylabel('Transmission Pressure Difference ')
        plt.subplot(212)
        plt.plot(self.expert[:, 2], 'r')
        plt.plot(self.expert[:, 4], 'm')
        plt.ylabel('Telescope Pressure')
        plt.xlabel('time')
        plt.show()


    def to_stages(self, y):
        n= len(y)
        stages = []
        samples = self.expert
        for i in range(n):
                if len(stages)<y[i]:
                    stages[y[i]].append(samples[i])
                else:
                    stages.append([samples[i]])

        for stage in stages:
            mu_and_sigma = self.get_mean_and_variance(np.array(stage))
            self.the_stages.append(mu_and_sigma)
        self.the_stages = np.array(self.the_stages)

    def reset(self):
        self.reward = 0

    def get_mean_and_variance(self, x):
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        return np.array([mu_x, sigma_x])

    def get_intermediate_reward(self, state):
        n = len(state)-1
        segment = self.clf.predict([state])[0]
        expert_t = self.the_stages[segment]
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
        if (time>300):
            self.reward -= time*100

        if segment>0:
            self.reward += reward_i*pow(2, segment-1)

    def clusters_to_segments(self, labels):
        prev_label = None
        i =0
        n_demo = 0
        for label in labels:
            if self.demonstrations[i][0] != n_demo:
                i = 0

            if prev_label==None:
               prev_label = label
            elif label != prev_label:
                i +=1
                prev_label = label
            self.y.append(i)

    def step_classifier(self):
        self.clf = SVC()
        self.clf.fit(self.X, self.y)

    def stages(self, n_clusters):

        self.X = self.expert
        cluster_centers = [self.expert[40], self.expert[120], self.expert[200]]
        cluster_centers = np.array(cluster_centers)
        clusters = KMeans(n_clusters=n_clusters, init=cluster_centers).fit(self.X)
        self.y = clusters.labels_
        # clusters = AgglomerativeClustering(n_clusters=n_clusters, )
        # y = clusters.fit_predict(self.X)
        # self.clusters_to_segments(y)
        plt.plot(self.y)
        plt.show()
        self.to_stages(self.y)