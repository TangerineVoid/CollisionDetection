# from universal_robot_kinematics import invKine
# import universal_robot_kinematics as urk
import numpy as np
import pandas as pd
from functions import *
from URKinematics import *
# READ DATA FROM FILE
# ------------------------------------------------------------------------
# READ DATA FROM FILE
# ------------------------------------------------------------------------
read_data = True
if read_data:
    data = read_data_from_file("results.txt")
    df = pd.DataFrame(data)
    indexes = np.array(df['index'].tolist())
    states = np.array(df['state'].tolist())
    angles = np.array(states[:,1])
    TCP = np.array(df['TCP'].tolist())

def IK_func(arr, js_old):
    # Inverse kinematics
    js = invKine(arr)
    # Obtain best js value
    js = MinMov(js, js_old)
    return js

def MinMov(js, js_old):
    js_old = np.repeat(js_old, repeats=js.shape[1], axis=1)
    distances = np.linalg.norm(js - js_old, axis=0)
    idx = np.argmin(distances)
    error = js[:,idx]
    return error

js_init = np.matrix([[-0.0229],[-1.1132],[0.4312],[2.2528],[-1.5708],[-3.1186]])
js = js_init
c = [0]
TCP[0,:3,3] = TCP[0,:3,3] + [-4.00016388e-01, -1.00016347e-01, 7.99987654e-01]
HT = TCP[0]
js = IK_func(HT, js_init)
js = js.transpose()