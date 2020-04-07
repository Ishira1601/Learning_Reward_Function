import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import k_means
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
import warnings
warnings.filterwarnings("ignore")

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
        self.stages(10)
        # self.clustering(3)
        self.step_classifier()


    def get_mean_std_expert(self):
        for file in self.files:
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                i = 0
                for row in csv_reader:
                    observation = [float(row[0]), float(row[1])]
                    if i>=len(self.demonstrations):
                        self.demonstrations.append([observation])
                    else:
                        self.demonstrations[i].append(observation)
                    i += 1
        for demonstrations_t in self.demonstrations:
            mu_and_sigma = self.get_mean_and_variance(np.array(demonstrations_t))
            self.expert.append(mu_and_sigma)
        self.expert = np.array(self.expert)
        plt.plot(self.expert[:, 0, 0], 'r')
        plt.plot(self.expert[:, 0, 1], 'b')
        plt.show()


    def to_stages(self, y):
        n= len(self.expert)
        stages = []
        samples = self.expert[:, 0, :]
        for i in range(n):
                if stages==[]:
                    stages.append([samples[i]])
                elif y[i]==y[i-1]:
                    stages[-1].append(samples[i])
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
        summed_min = np.inf
        i = 0
        for expert_t in self.the_stages:
            mu_t = expert_t[0]
            sigma_t = expert_t[1]
            summed = 0
            for j in range(2):
                    if sigma_t[j] is not 0.0:
                        summed = summed + (np.square(state[j] - mu_t[j])) / np.square(sigma_t[j])
                    else:
                        break
            if summed < summed_min and summed>0:
                summed_min = summed
                t = i
            i += 1
        # reward_t = summed_min
        reward_t = 100 - summed_min
        segment = self.clf.predict([[t, state[0], state[1]]])[0]
        return reward_t, t, segment

    def combine_reward(self, reward_i, segment, time):
        if (time>200):
            self.reward -= time*100

        if segment>0:
            self.reward += reward_i*pow(2, segment-1)

    def clusters_to_segments(self, labels):
        prev_label = None
        i =0
        for label in labels:
            if prev_label==None:
               prev_label = label
            if label != prev_label:
                i +=1
                prev_label = label
            self.y.append(i)

    def clustering(self, n_clusters):
        n_samples = self.the_stages.shape[0]
        n_features = self.the_stages.shape[2]+1
        time = np.arange(n_samples).reshape(n_samples, 1)
        samples = self.the_stages[:, 0, :]
        self.X = np.hstack((time, samples))
        # self.segments = AgglomerativeClustering(n_clusters=n_clusters, )
        centers = np.ndarray((n_clusters, n_features), buffer= np.array([[1, 0.47, 0.0099], [4, -0.6, -0.023], [9, -0.085, 0.035]]))
        self.segments = k_means(self.X, n_clusters=n_clusters, init=centers)
        self.y = self.segments[1]
        plt.plot(self.y)
        plt.show()

    def step_classifier(self):
        self.clf = SVC()
        self.clf.fit(self.X, self.y)

    def stages(self, n_clusters):
        clusters = AgglomerativeClustering(n_clusters=n_clusters, )
        n_time = self.expert.shape[0]
        time = np.arange(n_time).reshape(n_time, 1)
        samples = self.expert[:, 0, :]
        self.X = np.hstack((time, samples))
        y = clusters.fit_predict(self.X)
        self.clusters_to_segments(y)
        plt.plot(self.y)
        plt.show()
        self.to_stages(y)