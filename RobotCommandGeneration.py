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
read_coordinates = True
read_data = False
# file_name = "results_Canon.txt"
# file_name = "results_Triangle1.txt"
file_name = "results_Triangle3.txt"
# file_name = "results_Triangle3_paper.txt"
# file_name = "results_LongLine.txt"
# file_name = "results_4T3D.txt"
# file_name = "results_lattice.txt"
if read_data:
    data = read_data_from_file(file_name)
    df = pd.DataFrame(data)
    indexes = np.array(df['index'].tolist())
    states = np.array(df['state'].tolist())
    angles = np.array(states[:,1])
    TCP = np.array(df['TCP'].tolist())

# Obtain coordinates
if read_coordinates:
    data = np.array(read_data_from_file(file_name),float)

def IK_func(arr, js_old):
    # Inverse kinematics
    js = invKine(arr)
    # print(js)
    # Obtain best js value
    js = MinMov(js, js_old)
    return js

def MinMov(js, js_old):
    js_old = np.repeat(js_old, repeats=js.shape[1], axis=1)
    # distances = np.linalg.norm(js - js_old, axis=0)
    distances = np.linalg.norm(js[:3] - js_old[:3], axis=0)
    idx = np.argmin(distances)
    error = js[:,idx]
    return error

# Home test
# Sin el fixture del ventilador
# js_init = np.deg2rad(np.array([[111.76],[-163.36],[28.60],[44.78],[90],[-21.60]]))
# Para el fixture del ventilador:
js_init = np.deg2rad(np.array([[95.4],[-174],[58.4],[25.6],[90],[-47.5]]))
# FK_func(np.array(np.deg2rad([[-86.61],[-57.68],[52.2],[95.12],[-90.55],[-3.5]])),c)
js = js_init
c = [0]
js_commands = []
ps_init = np.array(FK_func(js_init,c))
ps_init = np.dot(ps_init[:3,:3],ps_init[:3,3])
ps = np.array([62,505.2,569.5])*1e-03
# ps = np.array([67.75,594,568.3])*1e-03
print(ps)
ps_commands = [ps]
# ------------------------------------
# If the given data are coordinates
# ------------------------------------
if read_coordinates:
    # ------------------------------------
    # Increase data resolution
    # ------------------------------------
    interpolated_data = []
    resolution = 10 # mm/s
    for i in range(len(data) - 1):
        start_command = data[i]
        end_command = data[i + 1]
        # Calculate the maximum number of steps needed for the largest movement
        max_steps = max(abs(end - start) for start, end in zip(start_command, end_command)) / resolution
        max_steps = int(math.ceil(max_steps))
        # Interpolate between the start and end commands
        for e,t in enumerate(np.linspace(0, 1, max_steps + 1)):
            intermediate_command = [start + t * (end - start) for start, end in zip(start_command, end_command)]
            intermediate_command.append(i)
            # # Avoid saving duplicated values between paths
            # if e <= max_steps-1 or i == range(len(data) - 1)[-1]:
            interpolated_data.append(intermediate_command)
    interpolated_data = np.array(interpolated_data)
    np.savetxt(f'resolution_{file_name}', np.array(interpolated_data)[:, :4], delimiter=" ")
    # ------------------------------------
    # Filter just the needed values
    # ------------------------------------
    f_interpolated_data = []
    for ip in range(0,int(interpolated_data[-1,3])+1):
        filter = np.argwhere(interpolated_data[:,3]==ip)
        filter = np.concatenate(filter).ravel()
        f_interpolated_data.append(filter[0])
    f_interpolated_data.append(len(interpolated_data)-1)
    # np.savetxt(f'resolution_{file_name}', np.array(interpolated_data)[f_interpolated_data, :4], delimiter=" ")

    # ------------------------------------
    # Get Inverse Kinematics
    # ------------------------------------
    data = interpolated_data[:,:3].copy()
    for d,dt in enumerate(range(0,len(data))):
        data[d,:] = data[d,:]*1e-03 + ps #+ np.array([0.0,0.0,173.40-172.36])*1e-03 #[1.452884000e-02, -5.714585640e-01, 7.062791910e-01]
        [angle_x,angle_y,angle_z] = [0,0,0]
        [tran_x, tran_y, tran_z] = [0,0,0]
        # Apply homogeneous transformation to coordinates
        data = HT(data.transpose(), [angle_x, angle_y, angle_z], [tran_x, tran_y, tran_z], [0, 0, 0], 1)[0]
        H_T = np.array(np.eye(4,4))
        H_T[:3,3] = data[d]
        js = np.array(IK_func(H_T, js))
        js_commands.append(js[:])
if read_data:
    for t,tcp in enumerate(range(0,len(TCP))):
        TCP[t,:3,3] = TCP[t,:3,3]*1e-03 + [-4.00016388e-01, -1.00016347e-01, 7.99987654e-01]
        HT = TCP[t]
        js = IK_func(HT, js)
        # js = js.transpose()
        # print(js.transpose())
        js_commands.append(js[:])
# ------------------------------------
# Display Forward Kinematics
# ------------------------------------
for j in np.array(js_commands):
    p = np.array(FK_func(j, c))
    p = np.dot(p[:3, :3], p[:3, 3])
    print(p)
# ------------------------------------
# Save File as a txt
# ------------------------------------
# np.savetxt(f'FKCommands_{file_name}', np.reshape(np.array(js_commands), (len(data),6))[np.array(f_interpolated_data),:], delimiter=" ")
np.savetxt(f'FKCommands_{file_name}', np.reshape(np.array(js_commands), (len(data),6))[:,:], delimiter=" ")