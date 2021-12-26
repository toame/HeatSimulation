import numpy as np
import re
import matplotlib
import math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import glob
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
#path = '/mnt/c/Users/tomo2/Desktop/python/Temperature10161837_013.txt'

#font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
#font_prop = FontProperties(fname=font_path)
#plt.rcParams['font.family'] = font_prop.get_name()
#plt.rc('font', family='Noto Sans CJK JP')
plt.rcParams["font.size"] = 16

# #7 (sign[1]=1,beta0=0.1,kappa0=1,mu2=1)
dataX = [512, 1024, 2048, 4096, 8192, 16384]
#dataY_20 = [29.224823,41.486502,55.277871,69.913360,86.568337,104.325440,125.290604, 150.885874,183.774589]
dataY_5 = [29.318715917127,41.555178848568,55.384394409503,70.227380223662,87.056091070878,105.034898518640,126.623029552538]
#dataY_2 = [29.083631,41.501339,55.019780767997,69.955390518267,87.024317829939,103.574132348593,126.209368358013, 147.392423,185.736016]
dataY_20 = [29.224823,41.486502,55.277871,69.913360,86.568337,104.325440,125.290604]
#dataY_2 = [29.083631,41.501339,55.235602,69.897912,86.697598,106.884027,121.809723]
#dataY_4 = [29.146109,41.567801,55.329227,69.592784,86.581534,106.090276,124.082675]
dataY_20 = get_alpha(dataY_20)
dataY_5 = get_alpha(dataY_5)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xscale("log")
plt.ylim([0.0,0.6])
ax.plot(dataX, dataY_5,markersize=6, label='0.5billion step',marker='o', linestyle='dashed')
ax.plot(dataX, dataY_20,markersize=6, label='2billion step',marker='^', linestyle='dashed')
#ax.plot(dataX, dataY2,label='Linear',linestyle="dashed")
ax.set_xlabel('num')
ax.set_ylabel('alpha')
plt.legend(loc='best')
#plt.title('$M = 200$', y = -0.3)
fig.savefig("kappa_alpha.png", dpi=200,bbox_inches="tight")
plt.close()