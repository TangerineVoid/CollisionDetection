import copy
import numpy as np
import open3d as o3d


class Environment:
    # This class represents an environment for reinforcement learning.
    def __init__(self, initial_state=None, process_index=0, last_index=0, index_array=None, trajectory=None,
                 geometry=None, dcgeometry=None, dclaser=None, dcfilament=None, vxmeltpool=None,
                 voxel_size=0.5,
                 translation=[0, 0, 0], action=np.array([0,0,0]).transpose(), prev_action=None,
                 TCP=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])):
        # vxgeometry=None,
        self.state = initial_state
        self.last_index = last_index
        self.process_index = process_index
        self.index_array = index_array
        self.trajectory = trajectory
        self.geometry = geometry
        # self.vxgeometry = vxgeometry
        self.voxel_size = voxel_size
        self.dcgeometry = dcgeometry

        self.translation = translation
        self.action = action
        self.prev_action = copy.deepcopy(self.action)

        self.mod_geometry = copy.deepcopy(self.geometry)
        self.mod_trajectory = copy.deepcopy(self.trajectory)
        # self.mod_vxgeometry = copy.deepcopy(self.vxgeometry)
        self.mod_dcgeometry = copy.deepcopy(self.dcgeometry)

        self.dclaser = dclaser
        self.dcfilament = dcfilament
        self.vxmeltpool = vxmeltpool
        self.TCP = TCP
        self.mod_TCP = copy.deepcopy(self.TCP)


    def step(self, action, ax):
        # Takes an action and returns the next state and reward. Args: action: An action chosen by the agent. Returns: A tuple of (next_state, reward).
        # Update the state based on the action taken
        # Register last collision state
        past_collision, past_collision_laser, past_collision_filament = self.check_Collision()
        # Identify which indexes have the desired process index
        # Select the last index so we select all the elements
        idx = np.where(self.index_array == self.process_index)[0][-1]
        # Define rotation
        if ax.upper() == 'X':
            Rot = [action, 0, 0]
        elif ax.upper() == 'Y':
            Rot = [0, action, 0]
        elif ax.upper() == 'Z':
            Rot = [0, 0, action]
        self.action = np.add(self.action,np.array(Rot).transpose())
        # Apply rotation to geometry, trajectory and TCP
        self.mod_geometry = self._HT(self.mod_geometry.transpose(), Rot, [0, 0, 0], [0, 0, 0], 1)[0]
        mod_geometry = self.mod_geometry[:idx, :3]
        self.mod_dcgeometry.points = o3d.utility.Vector3dVector(mod_geometry[:, :3])
        self.mod_trajectory = self._HT(self.mod_trajectory.transpose(), Rot, [0, 0, 0], [0, 0, 0], 1)[0]
        TCP = np.array([self.mod_TCP[:3, 3]]).transpose()
        TCP, Rotation_Matrix, _, _ = self._HT(TCP, Rot, [0, 0, 0], [0, 0, 0], 1)
        self.mod_TCP = self._updateTCP(TCP, Rotation_Matrix)
        # Define trajectory translation
        dx = self.mod_trajectory[self.process_index, 0] - self.mod_trajectory[self.last_index, 0]
        dy = self.mod_trajectory[self.process_index, 1] - self.mod_trajectory[self.last_index, 1]
        dz = self.mod_trajectory[self.process_index, 2] - self.mod_trajectory[self.last_index, 2]
        translation = [dx * -1, dy * -1, dz * -1]
        # COLLISION DETECTION
        # ------------------------------------------------------------------------
        collision, collision_laser, collision_filament = self.check_Collision()
        if collision == 1 and past_collision == collision and action == 0:
            reward = -1
        elif collision == 1:
            reward = -0.5
        elif collision == 0:
            reward = 1
        # If new collisions were created, the state is set to the previous state, negating any effect the agent's decision took into the environment
        # if collision_filament-past_collision_filament == 1:
        #   self.mod_geometry = (self._HT(self.mod_geometry.transpose(), [0, -action, 0], [0, 0, 0], [0, 0, 0], 1)[0])[:idx,:3]
        #   self.mod_trajectory = (self._HT(self.mod_trajectory.transpose(), [0, -action, 0], [0, 0, 0], [0, 0, 0], 1)[0])[:idx, :3]
        #
        #   TCP = np.array([self.mod_TCP[:3, 3]]).transpose()  # np.vstack([self.mod_TCP[:3,3],[0,0,0]])
        #   TCP, Rotation_Matrix, _, _ = self._HT(TCP, [0, -action, 0], [0, 0, 0], [0, 0, 0], 1)
        #   self.mod_TCP = self._updateTCP(TCP, Rotation_Matrix)
        #   self.mod_dcgeometry.points = o3d.utility.Vector3dVector(self.mod_geometry[:idx, :3])
        #   action = 0

        # Update state
        self.prev_action = np.add(self.prev_action, np.array(Rot).transpose())
        self.state = [collision, float(self.prev_action[1]), 0, 0, 0]
        if self.process_index > 0:
            self.state[2:] = translation
        return self.state, reward, collision

    def check_Collision(self):
        # COLLISION DETECTION
        # ------------------------------------------------------------------------
        # Generate Voxelized geometry
        self.mod_vxgeometry = o3d.geometry.VoxelGrid.create_from_point_cloud(self.mod_dcgeometry,                                                                          voxel_size=self.voxel_size)
        # Collisions between melt pool sphere and laser-filament
        collision_ccd_laser = np.array(self.vxmeltpool.check_if_included(self.dclaser.points))
        collision_ccd_filament = np.array(self.vxmeltpool.check_if_included(self.dcfilament.points))
        # Collisions between deposited geometry and laser-filament
        collision_pcd_laser = np.array(self.mod_vxgeometry.check_if_included(self.dclaser.points))
        collision_pcd_filament = np.array(self.mod_vxgeometry.check_if_included(self.dcfilament.points))
        # Identify collisions independent of the meltpool
        collision_laser = collision_pcd_laser ^ (collision_ccd_laser & collision_pcd_laser)
        collision_filament = collision_pcd_filament ^ (collision_ccd_filament & collision_pcd_filament)
        collision_laser = any(collision_laser)
        collision_filament = any(collision_filament)
        collision = int(collision_laser or collision_filament)
        return collision, int(collision_laser), int(collision_filament)

    def continue_process(self, step):
        if step != 0:
            # This is to avoid an issue when failing to find the answer in n steps in
            # the episode, this deformed the geometry and failed to find a collision
            self.last_index = self.process_index
            self.process_index = step + self.process_index
            if self.process_index < 0:
                self.process_index = 0
            if self.process_index > np.shape(self.mod_trajectory)[0]:
                self.process_index = np.shape(self.mod_trajectory)[0] - 1
                print("It has been reached the end of the process")
                # Update state
                self.state[0] = self.check_Collision()[0]
                if self.process_index > 0:
                    self.state[2:] = self.translation
                # Reset actions
                self.action = np.array([0, 0, 0]).transpose()
                print(f'continue process {step}, from {self.last_index} to {self.process_index}')
                return 0, self.state
            # Identify which indexes have the desired process index
            # Select the last index so we select all the elements
            idx = np.where(self.index_array == self.process_index)[0][-1]
            print(f'continue process {step}, from {self.last_index} to {self.process_index}')
            # Calculate translation
            dx = self.mod_trajectory[self.process_index, 0] - self.mod_trajectory[self.last_index, 0]
            dy = self.mod_trajectory[self.process_index, 1] - self.mod_trajectory[self.last_index, 1]
            dz = self.mod_trajectory[self.process_index, 2] - self.mod_trajectory[self.last_index, 2]
            translation = [dx * -1, dy * -1, dz * -1]
            self.translation = translation
            # Apply translations to geometry, trajectory and TCP
            self.mod_geometry = self._HT(self.mod_geometry.transpose(), [0, 0, 0], translation, [0, 0, 0], 1)[0]
            mod_geometry = self.mod_geometry[:idx, :3]
            self.mod_dcgeometry.points = o3d.utility.Vector3dVector(mod_geometry[:, :3])
            self.mod_trajectory = self._HT(self.mod_trajectory.transpose(), [0, 0, 0], translation, [0, 0, 0], 1)[0]
            TCP = np.array([self.mod_TCP[:3, 3]]).transpose()
            TCP, Rotation_Matrix, _, _ = self._HT(TCP, [0, 0, 0], translation, [0, 0, 0], 1)
            self.mod_TCP = self._updateTCP(TCP, Rotation_Matrix)
            # # Reset actions
            # self.action = np.array([0, 0, 0]).transpose()
            # Update state
            self.state = [self.check_Collision()[0], float(self.prev_action[1]), 0, 0, 0]
            if self.process_index > 0:
                self.state[2:] = self.translation
        return 1, self.state

    def reset(self):
        # Reset rotation transformations to set orientation of last deposition

        # Identify which indexes have the desired process index
        # Select the last index so we select all the elements
        idx = np.where(self.index_array == self.process_index)[0][-1]
        # Apply rotation to geometry, trajectory and TCP
        self.mod_geometry = self._HT(self.mod_geometry.transpose(), self.action*-1, [0, 0, 0], [0, 0, 0], 1)[0]
        mod_geometry = self.mod_geometry[:idx, :3]
        self.mod_dcgeometry.points = o3d.utility.Vector3dVector(mod_geometry[:, :3])
        self.mod_trajectory = self._HT(self.mod_trajectory.transpose(), self.action*-1, [0, 0, 0], [0, 0, 0], 1)[0]
        TCP = np.array([self.mod_TCP[:3, 3]]).transpose()
        TCP, Rotation_Matrix, _, _ = self._HT(TCP, self.action*-1, [0, 0, 0], [0, 0, 0], 1)
        self.mod_TCP = self._updateTCP(TCP, Rotation_Matrix)
        # Reset actions
        self.prev_action = self.prev_action - self.action
        self.action = np.array([0, 0, 0]).transpose()
        # Update state after reset
        self.state = [self.check_Collision()[0], float(self.prev_action[1]), 0, 0, 0]
        if self.process_index > 0:
            self.state[2:] = self.translation
        return self.state

    def _updateTCP(self, TCP, R):
        mod_TCP = np.zeros((4, 4))
        Rold = self.mod_TCP[:3, :3]
        mod_TCP[:3, :3] = np.dot(R, Rold)
        mod_TCP[:3, 3] = TCP
        mod_TCP[3, 3] = 1
        return mod_TCP

    def _HT(self, arr, rotation, translation, offset, scale):
        # Apply scaling and offset
        arr[:3, :] = arr[:3, :] * scale + np.array([offset]).transpose()
        arr = np.vstack([arr, np.ones(arr.shape[1])])
        # Create rotation matrices
        theta_x, theta_y, theta_z = np.radians(rotation)
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        # Combine rotation matrices
        R = np.dot(np.dot(Rx, Ry), Rz)
        # Construct homogeneous transformation matrix
        T = np.array(translation)
        H = np.vstack([np.hstack([R, T.reshape(-1, 1)]), np.array([0, 0, 0, scale])])
        # Apply transformation to arr
        arr = np.dot(H, arr).transpose()
        return arr[:, :3], R, T, H
