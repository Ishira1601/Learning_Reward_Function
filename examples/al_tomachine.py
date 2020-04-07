from irl.al import AL
import numpy as np

trajectories = np.zeros((16, 300, 82))

import csv

with open('18-54-34.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    i = 0
    for row in csv_reader:
        trajectories[0][i] = np.array(row)
        i+=1

print("done")
al = AL(n_actions= 4, discount =0.8, trajectories=trajectories, epochs =160, learning_rate=0.4, termination_threshold=0.01)
al.irl()