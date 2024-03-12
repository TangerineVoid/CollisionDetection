import re
import numpy as np
import io
from PIL import Image
import json
# Get coordinates from GCode file
# Get coordinates from GCode file
def parse_file(file_path,type):
    if type == 'gcode':
        # Regular expression pattern for extracting G, X, Y, and Z values
        # gcode_pattern = r'([GXYZ])(-?\d+\.?\d*)'
        gcode_pattern = r'([GXYZ])(-?\d+\.?\d*)|(-?\d+\.?\d*)([GXYZ])'
        # Lists to store parsed values
        g_values = []
        x_values = []
        y_values = []
        z_values = []
        # Read GCode data from the file
        with open(file_path, 'r') as file:
            gcode_lines = file.readlines()
        # Extract values using regex
        for line in gcode_lines:
            # print(re.findall(gcode_pattern,line.upper()))
            match = re.findall(gcode_pattern,line.upper())
            match = [tuple(filter(None, element)) for element in match]
            # match = gcode_pattern.match(line)
            if match:
                for e in match:
                    numbers = []
                    commands = []
                    for i,item in enumerate(e):
                        try:
                            float(item)
                            numbers.append(item)
                        except ValueError:
                            commands.append(item)
                    if commands[0]=='G':
                        g_values.append(float(numbers[0]))
                    if commands[0]=='X':
                        x_values.append(float(numbers[0]))
                    if commands[0]=='Y':
                        y_values.append(float(numbers[0]))
                    if commands[0]=='Z':
                        z_values.append(float(numbers[0]))
        center = [x_values[0]*np.ones(len(x_values)),y_values[0]*np.ones(len(y_values)),z_values[0]*np.ones(len(z_values))]
        commands = np.array([x_values-center[0],y_values-center[1],z_values-center[2]]).transpose()
    elif type == 'coordinates':
        x_values = []
        y_values = []
        z_values = []
        # Load data from the text file
        data = np.loadtxt(file_path, delimiter=",")
        center = np.array([data[0,0] * np.ones(len(data[:,0])), data[0,1] * np.ones(len(data[:,1])),data[0,2] * np.ones(len(data[:,2]))]).transpose()
        commands = np.array([data[:,0]-center[:,0],data[:,1]-center[:,1],data[:,2]-center[:,2]]).transpose()
    return commands

# Interpolate GCode coordinates to increase resolution
def extend_gcode(step, arr):
    ext_move = np.empty((0,3))
    for i in range(arr.shape[0] - 1):
        r = np.linalg.norm(arr[i + 1, :] - arr[i, :])
        number_step = r / step
        component_step = step * ((arr[i + 1, :] - arr[i, :]) / r)
        ext_move = np.append(ext_move,[arr[i, :]],axis=0)
        for j in range(int(np.floor(number_step))):
            ext_move = np.append(ext_move,[ext_move[-1] + component_step],axis=0)
        if i == arr.shape[0] - 2:
            ext_move = np.append(ext_move,[arr[i + 1, :]],axis=0)
    return ext_move

def generate_point_cloud_from_coordinates(coordinates, diameter, num_points):
    point_cloud = []

    for c, coord in enumerate(coordinates):
        x, y, z = coord
        if c > 0 and c < len(coordinates) - 1:
            v1 = coordinates[c+1]-coord
            v1 = v1/np.linalg.norm(v1)
            v2 = coord-coordinates[c-1]
            v2 = v2 / np.linalg.norm(v2)
            if np.all(v1 != (-1*v2)):
                vector = v1-(-1*v2)
            else:
                vector = v1
        elif c == 0:
            vector = coordinates[c+1]-coord
        elif c == len(coordinates):
             vector = coord-coordinates[c-1]
        normalized_vector = vector / np.linalg.norm(vector)
        # Calculate rotation angles
        angle_x = -np.degrees(np.arctan2(normalized_vector[1], normalized_vector[2]))
        angle_y = -np.degrees(np.arctan2(-normalized_vector[0],
                             np.sqrt(normalized_vector[1] ** 2 + normalized_vector[2] ** 2)))
        angle_z = np.degrees(np.arctan2(normalized_vector[0], normalized_vector[1]))
        # Generate base circle
        theta = np.linspace(0, 2 * np.pi, num_points)
        min_diam = 0.08
        num = 5
        diameters = np.linspace(min_diam, diameter, num)
        for d in diameters:
            # Generate circles
            circle_x, circle_y, circle_z = [(d / 2) * np.cos(theta), (d / 2) * np.sin(theta), np.full_like(np.cos(theta), 0)]
            # Combine the points into a single array
            circular_points = np.column_stack((circle_x, circle_y, circle_z)).transpose()

            # Homogeneous transformation
            circular_points = HT(circular_points, [angle_x, angle_y, angle_z], [x, y, z], [0, 0, 0], 1)[0]
            circular_points = np.column_stack((circular_points,np.full_like(circular_points[:,0],c)))
            point_cloud.append(circular_points)

    # Combine all circular point clouds into a single array
    point_cloud = np.concatenate(point_cloud)

    return point_cloud

def HT(arr, rotation, translation, offset, scale):
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
    return arr[:,:3], R, T, H

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def save_data2txt(filename, data):
    # data = data.to_json()
    with open(filename, "a") as file:
      json.dump(data, file)
      # for c in coordinates:
      file.write("\n")  # Write coordinates in two columns

def read_data_from_file(filename):
  data = []
  with open(filename, "r") as file:
    data_string = file.read()
    length = len(data_string.splitlines())
    for i, item in enumerate(data_string.split("\n")):  # Skip first line for clarity
        if i <= length -1:
          obj = json.loads(item)
          data.append(obj)
      # for line in file:
      #     # Split each line using ',' as the separator, and convert values to floats
      #     values = [value for value in line.strip().split(',')]
      #     data.append(values)
  return data


def generate_sphere_point_cloud(radius, point_density):
    num_concentric_spheres = 100
    # Generate spherical coordinates
    theta = np.linspace(0, 2 * np.pi, point_density)
    phi = np.linspace(0, np.pi, point_density)

    # Generate coordinates for all spheres
    x, y, z = [], [], []

    for i in range(num_concentric_spheres + 1):
        r = radius * i / num_concentric_spheres
        x.extend(r * np.outer(np.cos(theta), np.sin(phi)).flatten())
        y.extend(r * np.outer(np.sin(theta), np.sin(phi)).flatten())
        z.extend(r * np.outer(np.ones(np.size(theta)), np.cos(phi)).flatten())

    return np.array(x), np.array(y), np.array(z)