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

        # Find the amount of columns in data
        plt.xlabel("Number of threads")
        plt.ylabel("Relative speedup")
        headers = data.columns.values
        for i in range(1, len(data.columns)):
            name = data.columns[i]
            values = data[name].tolist()
            print("name is ", name)
            print("values are ", values)
            plt.plot(data[:,0], data[:,i], label=name)
        plt.legend()
        plt.show()
        break
            
            
            
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




