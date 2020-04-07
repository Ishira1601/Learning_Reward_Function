import numpy as np
from irl.upr import UPR
import gym
from q_learning import QAgent
import matplotlib.pyplot as plt

upr = UPR(['data_mc/demos.csv', 'data_mc/demos2.csv'])

s_i = np.array([0.4, 0.014])
r = upr.get_intermediate_reward(s_i)
print(r)

"""
Central Code Block.
"""
# Defining Gym Environment
env = gym.make("MountainCar-v0").env
state_size = 10

reward_function = []
steps_completed = []

def run_once():
    completed = False
    agent = QAgent(env, state_size)
    for i_episode in range(1000):
        reward_function = []
        upr.reset()
        observation= env.reset()
        done = False
        actions = []
        state = 0
        t = 0
        while not done:
            if (t>5000):
                # print("Nope", i_episode)
                break
            action = agent.get_action(state)
            next_observation, reward, done, info = env.step(action)

            reward_i, next_state, segment = upr.get_intermediate_reward(next_observation)

            if not reward_i==np.inf:
                # print(reward_i)
                upr.combine_reward(reward_i, segment, t)
            reward = upr.reward
            reward_function.append(reward)
            agent.train((state, action, next_state, reward, done))
            # env.render()

            state = next_state

            if done:
                print("Episode "+str(i_episode) + " finished in " + str(t) + " time steps!")
                print("R: ", upr.reward)
                print("epsilon: ", agent.eps)
                if t<160:
                    # plt.plot(reward_function)
                    # plt.ylabel('reward')
                    # plt.show()
                    steps_completed.append(i_episode)
                    completed = True
                break
            t += 1
        if completed:
            break

def run_multiple(n):
    for j in range(n):
        print("Starting over ", str(j))
        run_once()

    plt.plot(steps_completed)
    plt.ylabel('episodes to completion')
    plt.show()

run_once()
#run_multiple(100)
env.close()
