import numpy as np
from urx import Robot
from functions import *
from time import sleep
# pip install git+https://github.com/jkur/python-urx.git
# from functions import read_data_from_file

def move_robot_to_joint_positions(robot, joint_positions, v=0.055, t=0, r=0):
    # Move the robot to the specified joint positions
    # robot.movej(joint_positions, acc=3, vel=v) #0.5,0.7,----- 1.2
    robot.movej(joint_positions, acc=3, vel=v, t=t, r=r) #0.5,0.7,----- 1.2
    # robot.movej(joint_positions, acc=1, vel=0.25) #0.5,0.7,----- 1.2
    # robot.movej(joint_positions, acc=1, vel=0.5) #0.5,0.7,----- 1.2
    # robot.movej(joint_positions, acc=1, vel=1.5) #0.5,0.7,----- 1.2

def servo_move_robot_to_joint(robot, joint_positions, v=0.055, t=0.5, r=0, lookahead_time=0.03):
    # Move the robot to the specified joint positions
    # robot.movej(joint_positions, acc=3, vel=v) #0.5,0.7,----- 1.2
    robot.servoj(joint_positions,vel=0,acc=0,t=t,lookahead_time=lat,gain=500,wait=False) #+[0.01,0.01,0.01,0.01,0.01,0.01]
    # robot.movej(joint_positions, acc=1, vel=0.25) #0.5,0.7,----- 1.2
    # robot.movej(joint_positions, acc=1, vel=0.5) #0.5,0.7,----- 1.2
    # robot.movej(joint_positions, acc=1, vel=1.5) #0.5,0.7,----- 1.2

# def get_robot_position(robot)
#     robot.

if __name__ == "__main__":
    # Replace 'robot_ip' with the IP address of your UR robot
    robot_ip = "192.168.1.12"
    # file_name = "FKCommands_results_lattice.txt"
    # file_name = "FKCommands_results_Canon.txt"
    # file_name = "FKCommands_results_Triangle1.txt"
    file_name = "FKCommands_results_Triangle3.txt"
    # file_name = "FKCommands_results_Triangle3_paper.txt"
    # file_name = "FKCommands_results_LongLine.txt"
    # file_name = "FKCommands_results_4T3D.txt"
    jointCommands = np.loadtxt(file_name)
    positionCommands = np.loadtxt(f"resolution_{file_name.split('_',1)[1]}")
    #Identify the purge line
    purge_idx = [np.where(positionCommands[:,3]==0)[0][0], np.where(positionCommands[:,3]==0)[0][-1]]
    # positionCommands = np.array(read_data_from_file(file_name.split('_',1)[1]))
    # Create a UR robot object
    robot = Robot(robot_ip,use_rt=True)
    move_home = True
    lat = 0.008
    try:
        # Example joint positions (replace with your desired joint values)
        for e,pc in enumerate(positionCommands):
            # print(e)
            js = jointCommands[e]
            # js = np.array([-0.0229,-1.1132,0.4312,2.2528,-1.5708,-3.1186])
            # Move the robot to the specified joint positions
            # print(js.tolist())
            js[-1] = np.deg2rad(-47.54)
            if e > 0:
                # Define trajectory speed
                v = 25  # mm/s
                d = np.linalg.norm(positionCommands[e,:3]-positionCommands[e-1,:3])
                t = d/v
                if positionCommands[e,2]-positionCommands[e-1,2] > 0:
                    v = 25 # mm/s
                    # d = np.linalg.norm(positionCommands[e, :3] - positionCommands[e - 1, :3])
                    t = d / v
                if e == 0: # Go Home
                    move_robot_to_joint_positions(robot, js.tolist(), 0.1)
                elif e <= purge_idx[1]+1: # Purge Line
                    # move_robot_to_joint_positions(robot, js.tolist(), v=0.1)
                    v = 10  # mm/s
                    t = d / v
                    servo_move_robot_to_joint(robot,js.tolist(),t=t,lookahead_time=t/2)
                else:
                    # move_robot_to_joint_positions(robot, js.tolist(),t=t)
                    servo_move_robot_to_joint(robot,js.tolist(),t=t,lookahead_time=t/2)
                if t!=0: sleep(t-t/12)
                print(t,t-t/18)
        if move_home:
            js = jointCommands[0]
            js[-1] = np.deg2rad(-47.54)
            move_robot_to_joint_positions(robot, js.tolist(), 0.1)
            # servo_move_robot_to_joint(robot,js.tolist())
    finally:
        # Close the connection to the robot when done
        robot.close()
        print("Connection closed.")

#
# rob = urx.Robot("192.168.0.100")
# rob.set_tcp((0, 0, 0.1, 0, 0, 0))
# rob.set_payload(2, (0, 0, 0.1))
# sleep(0.2)  #leave some time to robot to process the setup commands
# rob.movej((1, 2, 3, 4, 5, 6), a, v)
# rob.movel((x, y, z, rx, ry, rz), a, v)
# print "Current tool pose is: ",  rob.getl()
# rob.movel((0.1, 0, 0, 0, 0, 0), a, v, relative=true)  # move relative to current pose
# rob.translate((0.1, 0, 0), a, v)  #move tool and keep orientation
# rob.stopj(a)
#
# rob.movel(x, y, z, rx, ry, rz), wait=False)
# while True :
#     sleep(0.1)  #sleep first since the robot may not have processed the command yet
#     if rob.is_program_running():
#         break
#
# rob.movel(x, y, z, rx, ry, rz), wait=False)
# while rob.getForce() < 50:
#     sleep(0.01)
#     if not rob.is_program_running():
#         break
# rob.stopl()
#
# try:
#     rob.movel((0,0,0.1,0,0,0), relative=True)
# except RobotError, ex:
#     print("Robot could not execute move (emergency stop for example), do something", ex)