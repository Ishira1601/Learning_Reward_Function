from irl.al import AL
import numpy as np
import numpy.random as rn
trajectories = np.zeros((2, 5, 2))

def create_grid(n):
    grid = []

    for i in range(n):
        for j in range(n):
            grid.append((i, j))
    return grid

def is_neighbor(a, b):
    if (abs(a[0]-b[0])==1 or abs(a[1]-b[1])==1):
        return True

def create_graph(grid):
    graph = np.zeros((len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            if (is_neighbor(grid[i], grid[j])):
                graph[i, j] = 1

grid = create_grid(3)
start = 0
goal = 8
current=0
path = [start]
next = 0
while (current!=goal):
    while (not is_neighbor(grid[next], grid[current])):
        next = rn.randint(0, len(grid))
    path.append(next)
    current = next
print("done")