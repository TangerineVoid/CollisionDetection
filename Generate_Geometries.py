import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the spiral
theta = np.linspace(0, 4*np.pi, 200)  # Angle range
a = 1  # Radius factor
b = 1  # Height factor

# Calculate x, y, and z coordinates
x = a * theta * np.cos(theta)
y = a * theta * np.sin(theta)
z = b * theta

# Save coordinates to a text file
data = np.stack((x, y, z), axis=1)  # Combine coordinates into a 2D array
np.savetxt("spiral_coordinates.txt", data, delimiter=",")  # Save with comma delimiter

# Create the plot (optional)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='Spiral')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Spiral')

plt.legend()
plt.show()
