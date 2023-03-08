# Read temp.csv and plot the data
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


# get path of current file
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path = path + "/perf-test/"

# Read each csv file from the benchmarks/perf-test directory
for file in os.listdir(path):
    if file.endswith(".csv"):

        file_path = path + file
        print(file_path)

        # The above causes nans due to f64, instead use
        # data = np.genfromtxt(file_path, delimiter=',', dtype=None)
        # Read the file 
        data = pd.read_csv(file_path, delimiter=',')
        break

        # # Find the amount of columns in data
        # columns = data.shape[1]
        # for i in range(0, columns):
        #     # read header row, which is the first row from data
        #     header = data[0,:]
        #     print(header)
            
            
            # Plot the data
            # plt.plot(data[:,0], data[:,i], label="Thread " + str(i+1))
            # plt.xlabel("Number of threads")
            # plt.ylabel("Relative speedup")
            # plt.legend()
            # plt.show()

        # Plot second tuple values against third tuple values, x axis is the number of threads
        # plt.plot(data[:,0], data[:,1], label="one shot")
        # plt.plot(data[:,0], data[:,2], label="run for")
        # plt.xlabel("Number of threads")
        # plt.ylabel("Relative speedup")
        # plt.legend()
        # plt.show()




