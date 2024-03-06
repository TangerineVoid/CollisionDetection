from functions import *
import open3d as o3d
from environment import *

# REPRESENT PHYSICAL SYSTEM
# ------------------------------------------------------------------------
# Workpiece Representation
# ------------------------------------------------------------------------
# Specify the path to your GCode file
# gcode_file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Triangle4_C.txt"
gcode_file_path =  "C:/Users/salin/Documents/Doctorado/LaserOcclusion/UnitLattices_HalfBCC.txt"

# Obtain coordinates
_,points = parse_gcode_file(gcode_file_path)
# Transform coordinates if needed
points = HT(points.transpose(), [0,0,0], [0,0,0], [0,0,0], 1)[0]
# Increase coordinates resolution
step_value = 0.1
points[:,2] = points[:,2]*-1
ext_trajectory= extend_gcode(step_value, points)
ext_points = generate_point_cloud_from_coordinates(ext_trajectory,1,90)

# initialize geometry pointcloud, and voxel instance
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(ext_points[:,:3]))
pcd.paint_uniform_color(np.array([0, 128, 255])/255)
voxel_size = 0.1

# ------------------------------------------------------------------------
# Laser Representation
# ------------------------------------------------------------------------
# initialize laser pointcloud instance
laser = np.array([[0,0,0],[-10,0,5]])
ext_laser= extend_gcode(np.linalg.norm(laser[1] - laser[0])/50, laser)
ext_laser = generate_point_cloud_from_coordinates(ext_laser,0.1,90)
lcd = o3d.geometry.PointCloud()
# ------------------------------------------------------------------------
lcd.points = o3d.utility.Vector3dVector(np.array(ext_laser[:,:3]))
lcd.paint_uniform_color(np.array([255, 0, 0])/255)

# ------------------------------------------------------------------------
# Filament Representation
# ------------------------------------------------------------------------
# initialize filament pointcloud instance
filament = np.array([[0,0,0],[-10,0,15]])
ext_filament= extend_gcode(np.linalg.norm(filament[1] - filament[0])/50, filament)
ext_filament = generate_point_cloud_from_coordinates(ext_filament,1,90)
fcd = o3d.geometry.PointCloud()
# ------------------------------------------------------------------------
fcd.points = o3d.utility.Vector3dVector(np.array(ext_filament[:,:3]))
fcd.paint_uniform_color(np.array([0, 0, 153])/255)

# ------------------------------------------------------------------------
# Meltpool Representation (Collision Sphere)
# ------------------------------------------------------------------------
radius = 0.75
point_density = 200
x, y, z = generate_sphere_point_cloud(radius, point_density)
# initialize geometry pointcloud, and voxel instance
ccd = o3d.geometry.PointCloud()
ccd.points = o3d.utility.Vector3dVector(np.transpose(np.array([x[:],y[:],z[:]])))
ccd.paint_uniform_color(np.array([255, 128, 0])/255)
# ------------------------------------------------------------------------
# Specify the voxel size
voxel_size = 0.1
ccd_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(ccd, voxel_size=voxel_size)

# ------------------------------------------------------------------------
# Create environment
# ------------------------------------------------------------------------
env = Environment()
env.index_array = ext_points[:,3]
env.process_index = 0
# Identify which indexes have the desired process index to select the segment of workpiece
idx = np.where(env.index_array == env.process_index)[0]
# Select the last index so we select all the elements
idx = idx[-1]
env.trajectory = ext_trajectory
env.mod_trajectory = ext_trajectory
env.geometry = np.asarray(pcd.points)
env.mod_geometry = np.asarray(pcd.points)
# env.vxgeometry = voxel_grid
env.dcgeometry = pcd
# env.mod_dcgeometry = pcd
env.mod_dcgeometry = env.dcgeometry
# self.mod_geometry = self._HT(self.mod_geometry.transpose(), [0,action,0], [0,0,0], [0,0,0], 1)[0]
# mod_geometry =
env.mod_dcgeometry.points = o3d.utility.Vector3dVector((env.mod_geometry[:idx, :3])[:,:3])
# mod_geometry = self.mod_geometry[:idx, :3]
# self.mod_dcgeometry.points = o3d.utility.Vector3dVector(mod_geometry[:, :3])
env.voxel_size = voxel_size
env.dclaser = lcd
env.dcfilament = fcd
env.vxmeltpool = ccd_voxel_grid
env.state = [0,0]