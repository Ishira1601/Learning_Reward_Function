import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import k_means
from sklearn.svm import SVC
from keras import Input, Sequential, Model,regularizers
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Lambda
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import cv2
from keras.optimizers import Adam

class UPR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.observations = []
        self.expert_segments = []
        self.reward = 0
        self.demonstrations = []
        self.expert = []
        self.n_segments = 10
        self.load_date()
        self.clustering(self.n_segments)
        self.set_model()

    def load_date(self):
        self.observations = []  # Images go here
        for i in range(0, 300):
            # Load image and parse class name
            img = cv2.imread(self.image_path + "/%d.jpg" % i, 0)
            self.observations.append(img)

        self.observations = np.array(self.observations)

    def clustering(self, n_clusters):
        clusters = AgglomerativeClustering(n_clusters=n_clusters, )
        n_time = self.observations.shape[0]
        r = self.observations.shape[1]
        c = self.observations.shape[2]
        n_features = r*c
        time = np.arange(n_time).reshape(n_time, 1)
        self.samples = self.observations.reshape(n_time, n_features)
        X = np.hstack((time, self.samples))
        y = clusters.fit_predict(X)

        self.clusters_to_X(y)
    def clusters_to_X(self, y):
        n = len(y)
        for i in range(1, n):
            if y[i-1]==y[i]:
                continue
            else:
                self.expert.append(self.observations[i-1])

        self.expert.append(self.observations[-1])
        self.expert = np.array(self.expert)
        print(self.expert)

    def set_model(self):
        r = self.observations.shape[1]
        c = self.observations.shape[2]

        self.model = get_siamese_model((r, c, 1))
        optimizer = Adam(lr=0.00006)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer)

    def get_intermediate_reward(self, observation):
        observation_rep = np.matlib.repmat(observation, self.n_segments)
        inputs = [observation_rep, self.expert]
        probs = self.model.predict(inputs)

def get_siamese_model(input_shape):
    """
        Model architecture
    """
    def initialize_weights():
        return 'glorot_uniform'

    def initialize_bias():
        return 'zeros'

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights(), kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights(),
                     bias_initializer=initialize_bias(), kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights(),
                     bias_initializer=initialize_bias(), kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights(),
                     bias_initializer=initialize_bias(), kernel_regularizer=regularizers.l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net
