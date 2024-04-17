from init import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import random

def solve_random(iterations):
    count = 0
    run_time = time.time()
    while count < iterations:
        print(f"Solving iteration {count}")
        # Initialize variables
        ext_trajectory,env,file_path = initialize()[:3]
        # Enable/Disable Save Results
        save_txt = True
        save_csv = True
        # Define number of inputs, and actions
        num_inputs = 5  # state, one angle, and the displacement vector
        num_actions = 3
        num_hidden = 128
        episode_count = 0
        process_index = env.process_index
        state = env.state
        process_step = 20
        angle_step = 10

        # NORMALIZATION VARIABLES
        min_x, min_y, min_z = [min(ext_trajectory[:, 0]), min(ext_trajectory[:, 1]), min(ext_trajectory[:, 2])]
        max_x, max_y, max_z = [max(ext_trajectory[:, 0]), max(ext_trajectory[:, 1]), max(ext_trajectory[:, 2])]
        rx, ry, rz = np.full_like([0,0,0], np.linalg.norm([np.linalg.norm([min_x,max_x]),
                                                           np.linalg.norm([min_y,max_y]),
                                                           np.linalg.norm([min_z,max_z])]))
        file_name = file_path.split("/")[-1].split(".")[0]
        date = time.strftime("%Y-%m-%d_%H-%M-%S")

        while True:  # Run until solved
            if episode_count > 0 and collision == 1:
                state = env.tool_reset()
            rt = 1
            timestep = 0
            while True:
                    norm_state = state
                    # Avoid changing angle if there is no collision
                    if state[0] == 0:
                        action = 2
                    else:
                        # Sample action from action probability distribution
                        action = random.randint(0,2)
                    if action == 2:
                      angle_change = 0
                    elif action == 1:
                      angle_change = -angle_step
                    elif action == 0:
                      angle_change = angle_step
                    # angle = angle_change
                    # angletotal = angletotal + angle
                    start_time = time.time()
                    # state, reward, collision = env.step(angle_change,'Y')
                    state, reward, collision = env.tool_step(angle_change,'Y')
                    if save_txt:
                        data = {
                            "ran time": time.time() - start_time,
                            "workpiece size": np.array(env.mod_dcgeometry.points).nbytes,
                            "episode": episode_count,
                            "step": timestep,
                            "index": process_index,
                            "action": action,
                            "collision": collision,
                            "reward": reward,
                            "state": np.array(norm_state).tolist(),#state,
                            "TCP": env.mod_TCP.reshape((4, 4)).tolist()  # env.mod_trajectory[0,:].tolist()
                        }
                        save_data2txt(f"Training_{file_name}_RL-AC_{num_inputs}S{num_actions}A_{date}.txt", data)
                    # Collision condition
                    if collision == 0:
                      break
                    timestep+=1
            # Continue next step in process
            # Analyze next step in process
            if collision == 0:
                # rt,state = env.continue_process(process_step)
                rt,state = env.tool_continue_process(process_step)
                if save_txt:
                    data = {
                        "episode": episode_count,
                        "step": timestep,
                        "index": process_index,
                        "action": action,
                        "collision": collision,
                        "reward": reward,
                        "state": np.array(norm_state).tolist(),#state,
                        "TCP": env.mod_TCP.reshape((4, 4)).tolist()  # env.mod_trajectory[0,:].tolist()
                    }
                    process_index = env.process_index
                    save_data2txt(f"Results_{file_name}_RL-AC_{num_inputs}S{num_actions}A_{date}.txt", data)
                # print("disque no hay colision")
            else:
                # env.continue_process(0)
                pass
            if rt == 0:
                count+=1
                break
            if save_csv:
                save_data2csv(f"Training_{file_name}_RL-AC_{num_inputs}S{num_actions}A_{date}.txt")
            episode_count += 1
        # print(f'Training experiment run time = {time.time() - run_time}')
    print(f'Total training run time = {time.time()-run_time}')

if __name__ == "__main__":
    solve_random(30)
