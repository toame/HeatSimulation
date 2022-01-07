import numpy as np
import re
import matplotlib
import math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression

mu0 = 1

def calc_avg(data, count):
    data = np.array(data)
    return data.reshape(count, -1).mean(axis=1)

path = "data/20211226" + str(mu0)
print(path)
setting_files = glob.glob(path + "/**/Setting.txt")
modalFlux050_files = glob.glob(path + "/**/modalFlux_050.txt")
modalFlux100_files = glob.glob(path + "/**/modalFlux_100.txt")
print(setting_files)
model_size_list = []
dataX = []
dataY = []
for file in setting_files:
    df = pd.read_csv(file, header=None)
    #print(df)
    model_size = int(df.iat[0, 1] + 0.5)
    print(model_size)
    model_size_list.append(model_size)

for file in modalFlux100_files:
    df = pd.read_csv(file, header=None)
    #print(df)
    #kappa = (df.iat[-1, 1] * df.iat[-1, 2] - df.iat[lower, 1] * df.iat[lower, 2]) / (df.iat[-1, 2] - df.iat[lower, 2])
    dataX.append(df[1])
    dataY.append(df[2])
zip_data = zip(model_size_list, dataX, dataY)
zip_sort = sorted(zip_data)

fig = plt.figure()
ax = fig.add_subplot(111)
for model_size, X, Y in zip_sort:
    X_ave = calc_avg(X, min(model_size//2, 1024))
    Y_ave = calc_avg(Y, min(model_size//2, 1024))
    ax.plot(X_ave, Y_ave, label='$N = ' + str(model_size) + '$',marker='None')
    plt.legend(loc='best')
fig.savefig("modal_flux" + str(mu0) + ".png", dpi=200,bbox_inches="tight")
plt.close()

for model_size, X, Y in zip_sort:
    for i in range(len(Y)):
        if(Y[i] < Y[i + 1] and Y[i + 1]> Y[i + 2]):
            print(X[i + 1], Y[i + 1])
            break
