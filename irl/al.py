"""
Implements apprentice learning (Abbeel and Ng 2004)

Ishira Liyanage
ishira.dewundaraliyanage@tuni.fi
"""

import numpy as np
import numpy.random as rn

class AL:
    def __init__(self, n_actions, discount,
        trajectories, epochs, learning_rate, termination_threshold):
        self.n_states, self.d_states = trajectories[0].shape
        self.discount = discount
        self.trajectories = trajectories

        self.termination_threshold = termination_threshold

        self.epochs = epochs

    def irl(self):
        # Find expert's feature expectation
        self.feature_expectation_expert = self.find_feature_expectation(self.trajectories)

        # Find random policies feature_expectations
        i = rn.randint(0, self.trajectories.shape[0])
        policy_0= self.trajectories[i]
        feature_expectation_policies = []
        feature_expectation_policies.append(self.find_feature_expectation([policy_0]))

        i = 0
        t = -1
        r = np.zeros((self.n_states,))
        while (t<self.termination_threshold):
            for e in range(self.epochs):
                w = rn.uniform(size=(self.d_states,))
                minimum = 100
                for feature_expectation_policy in feature_expectation_policies:
                    V = np.dot(w, (self.feature_expectation_expert - feature_expectation_policy))
                    if V<minimum:
                        minimum = V
                if minimum>t:
                    t = minimum
                    self.w = w

                r = self.w*self.trajectories[i]
        return r

    def rl(self, actions, r):
        Q = np.zeros((self.n_state, self.n_actions))
        state = 0
        for e in range(self.epochs):
            action = rn.choice(0, self.n_actions, actions)
            next_state = get_state(action)
            next_best_action = np.argmax(Q[next_state])
            Q[state][action] = r[state] + Q[next_state][next_best_action]
            state = next_state

    def find_feature_expectation(self, trajectories):
        feature_expectations = np.zeros(self.d_states)
        for feature_matrix in trajectories:
            for feature_vector in feature_matrix:
                feature_expectations += self.discount*feature_vector

        return feature_expectations


def get_state(action):
    state = 1
    return state


