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
def get_alpha(data_Y):
    
    ret = []
    data_num = 2
    for i in range(len(data_Y) - data_num + 1):
        retX = []
        retY = []
        for k in range(data_num):
            retX.append([i + k])
            retY.append(math.log(data_Y[i + k], 2))
        model = LinearRegression()
        model.fit(retX, retY)
        print('切片:', model.intercept_)
        print('傾き:', model.coef_)
        ret.append(model.coef_[0])

    return ret
def generate_line(dataX, dataY, coef_):
    #print(coef_)
    retX = [dataX[0], dataX[-1] * 2]
    diffX = math.log(retX[-1]/retX[0], 2)
    #print(diffX)
    retY = [dataY[0], dataY[0] * pow(2, coef_ * diffX)]
    print(retX, retY)
    print(retY[-1] * pow(2, -1/3))
    return (retX, retY)


for mu0 in ["0", "1"]:
    path = "data/20211226" + str(mu0)
    print(path)
    setting_files = glob.glob(path + "/**/Setting.txt")
    kappa_files = glob.glob(path + "/**/kappa.txt")
    print(setting_files)
    dataX = []
    dataY = []
    for file in setting_files:
        df = pd.read_csv(file, header=None)
        #print(df)
        model_size = int(df.iat[0, 1] + 0.5)
        print(model_size)
        dataX.append(model_size)
    lower = 293
    for file in kappa_files:
        df = pd.read_csv(file, header=None)
        #print(df)
        kappa = (df.iat[-1, 1] * df.iat[-1, 2] - df.iat[lower, 1] * df.iat[lower, 2]) / (df.iat[-1, 2] - df.iat[lower, 2])
        print(kappa)
        dataY.append(kappa)

    dataX.sort()
    dataY.sort()
    print(dataX, dataY)
    linearX1, linearY1 = generate_line(dataX, dataY, 1/3)
    linearX2, linearY2 = generate_line(dataX, dataY, 2/5)
    plt.rcParams["font.size"] = 16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dataX, dataY,markersize=6, label='2billion step',marker='o', linestyle='dashed')
    ax.plot(linearX1, linearY1,markersize=6, label='alpha = 1/3',marker='', linestyle='dashed')
    ax.plot(linearX2, linearY2,markersize=6, label='alpha = 2/5',marker='', linestyle='dashed')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='best')
    fig.savefig("kappa" + mu0 + ".png", dpi=200,bbox_inches="tight")
    plt.close()