import numpy as np
from urx import Robot
# pip install git+https://github.com/jkur/python-urx.git
# from functions import read_data_from_file

def move_robot_to_joint_positions(robot, joint_positions):
    # Move the robot to the specified joint positions
    robot.movej(joint_positions, acc=0.5, vel=0.1)

if __name__ == "__main__":
    # Replace 'robot_ip' with the IP address of your UR robot
    robot_ip = "192.168.1.12"
    jointCommands = np.loadtxt("FKCommands.txt")
    # Create a UR robot object
    robot = Robot(robot_ip)
    try:
        # Example joint positions (replace with your desired joint values)
        for js in jointCommands:
            # js = np.array([-0.0229,-1.1132,0.4312,2.2528,-1.5708,-3.1186])
            # Move the robot to the specified joint positions
            print(js.tolist())
            move_robot_to_joint_positions(robot, js.tolist())

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