# Read temp.csv and plot the data
import numpy as np

data = np.genfromtxt("benchmarks/perf-test/temp.csv", delimiter=',')
print(data)
# Data is a list of tuples, with the first element being the number of threads, and the second element being the time taken for one shot, the third element being the time taken for a given period of time
# Plot this data as a graph

import matplotlib.pyplot as plt

# Plot second tuple values against third tuple values, x axis is the number of threads
plt.plot(data[:,0], data[:,1], label="one shot")
plt.plot(data[:,0], data[:,2], label="run for")
plt.xlabel("Number of threads")
plt.ylabel("Relative speedup")
plt.legend()
plt.show()
