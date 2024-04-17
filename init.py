from functions import *
from environment import *
# import open3d as o3d

def initialize():
    # REPRESENT PHYSICAL SYSTEM
    # ------------------------------------------------------------------------
    # Workpiece Representation
    # ------------------------------------------------------------------------
    # Specify the path to your GCode file
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Triangle_C.txt"
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Triangle4_C.txt"
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Star.txt"
    file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/HourGlass.txt"
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Square.txt"
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/spiral_coordinates.txt"
    # file_path = "C:/Users/salin/Documents/Doctorado/LaserOcclusion/Truss_HalfBCC.txt"
    # file_path =  "C:/Users/salin/Documents/Doctorado/LaserOcclusion/UnitLattices_HalfBCC.txt"

    # Obtain coordinates
    points = parse_file(file_path,'gcode')
    # points = parse_file(file_path,'coordinates')
    # Transform coordinates if needed
    points = HT(points.transpose(), [0,0,0], [0,0,0], [0,0,0], 1)[0]
    # Increase coordinates resolution
    step_value = 0.1 #mm
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
    laser = np.array([[0,0,0],[-10,0,3.63970234266202]])#np.array([[0,0,0],[-10,0,5]])
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
    env.index_array = ext_points[:,3] # Set group of indexes to select sections of geometry
    env.process_index = 0 # Set starting point on simulation
    # env.process_index = 1600 # Set starting point on simulation
    # ------ Set trajectory data
    env.trajectory = ext_trajectory # Set trajectory
    env.mod_trajectory = env.trajectory # Set trajectory variable to be modified
    # ------ Set geometry coordinates data
    env.geometry = np.asarray(pcd.points) # Set geometry coordinate points
    env.mod_geometry = np.asarray(pcd.points) # Set geometry coordinate points to be modified
    # ------ Set geometry data cloud data
    env.dcgeometry = pcd # Set geometry data cloud
    env.mod_dcgeometry = env.dcgeometry # Set geometry data cloud to be modified
    # Identify which indexes have the desired process index to select the segment of workpiece
    idx = np.where(env.index_array == env.process_index)[0]
    # Select the last index so we select all the elements
    idx = idx[-1]
    env.mod_dcgeometry.points = o3d.utility.Vector3dVector((env.mod_geometry[:idx, :3])[:,:3]) # Select portion of geometry to be shown depending on the starting index
    # ------ Set laser data
    env.dclaser = lcd # Set laser data cloud
    if env.process_index != 0:
        points = HT(np.array(lcd.points).transpose(), [0, 0, 0], ext_trajectory[env.process_index], [0, 0, 0], 1)[0]
        lcd.points = o3d.utility.Vector3dVector(points)
    env.mod_laser = np.asarray(lcd.points) # Set laser data cloud to be modified
    env.voxel_size = voxel_size
    # ------ Set filament data
    env.dcfilament = fcd # Set laser data cloud
    if env.process_index != 0:
        points = HT(np.array(fcd.points).transpose(), [0, 0, 0], ext_trajectory[env.process_index], [0, 0, 0], 1)[0]
        fcd.points = o3d.utility.Vector3dVector(points)
    env.mod_filament = np.asarray(fcd.points) # Set filament data cloud to be modified
    # ------ Set meltpool data
    env.dcmeltpool = ccd # Set meltpool data cloud
    if env.process_index != 0:
        points = HT(np.array(ccd.points).transpose(), [0, 0, 0], ext_trajectory[env.process_index], [0, 0, 0], 1)[0]
        ccd.points = o3d.utility.Vector3dVector(points)
    env.mod_dcmeltpool = env.dcmeltpool # Set meltpool data cloud to be modified
    # ------ Check starting state
    env.state = [env.check_Collision()[0], env.action[1], 0, 0, 0]

    return ext_trajectory,env,file_path,lcd,fcd