# Read temp.csv and plot the data
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path = path + "/perf-test/"

for file in os.listdir(path):
    if file.startswith("run") and file.endswith(".csv"):
        file_path = path + file
        print(file_path)

        data = pd.read_csv(file_path, delimiter=",")

        plt.style.use("dark_background")
        plt.title(file)
        plt.xlabel("Number of threads")
        plt.ylabel("Relative speedup")

        headers = data.columns.values

        for i in range(1, len(data.columns)):
            name = headers[i]
            values = data[headers[i]].tolist()
            plt.plot(data[headers[0]].tolist(), values, label=name)

        plt.legend()
        plt.grid(which="major", color="#666666", linestyle="-")
        plt.minorticks_on()

        plt.show()
        file_path_without_suffix = os.path.splitext(file_path)[0]
        plt.savefig(file_path_without_suffix + ".png")
