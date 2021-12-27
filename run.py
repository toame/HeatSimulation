import os
import subprocess
import datetime

SLASH = ""
if os.name == 'nt':
    SLASH = '\\'
elif os.name == 'posix':
    SLASH = '/'
now = datetime.datetime.now()
ID = '{0:%m%d%H%M}'.format(now)
for i in range(10, 21):
    for mu1 in [0, 1]:
        ID = "20211226" + str(mu1)
        dir_path = 'data' + SLASH + ID + SLASH + 'N=' + str(2**i)
        os.makedirs(dir_path, exist_ok=True)
        allSteps = 2000000000
        if i == 19:
            allSteps = 5000000000
        elif i == 20:
            allSteps = 10000000000
        cmd  = 'nvcc -lcufft -o Molecular_dynamics_cuda Molecular_dynamics_cuda.cu -O2'
        cmd += ' -DSIZE=' + str(2**i)
        cmd += ' -Ddt=0.05'
        cmd += ' -Dmu1=' + str(mu1)
        cmd += ' -DallSteps=' + str(allSteps)
        cmd += ' -DinitialStateSteps=10000000'
        cmd += ' -DinitialHeatSteps=500000'
        cmd += ' -DID=' + ID

        returncode = subprocess.call(cmd, shell=True)
        if os.name == 'nt':
            returncode = subprocess.call('Molecular_dynamics_cuda', shell=True)
        elif os.name == 'posix':
            returncode = subprocess.call('./Molecular_dynamics_cuda', shell=True)
