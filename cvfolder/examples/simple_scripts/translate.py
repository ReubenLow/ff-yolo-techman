import numpy as np
from scipy.spatial.transform import Rotation as R

# Hypothetical corresponding points (in reality, you would gather these experimentally)
# Points in the camera coordinate system (X_c, Y_c, Z_c)
camera_points = np.array([
    [7.33, -22.24, 313.00],
    [90.22, -13.36, 279.00],
    [-57.03, -7.06, 225.00]
])

# Corresponding points in the robot's base coordinate system (X_r, Y_r, Z_r)
robot_points = np.array([
    [-111.20, -564.22, 63.55],
    [-18.62, -513.52, 70.94],
    [-89.08, -662.95, 62.49]
])

# Center the points (subtract the mean)
camera_mean = np.mean(camera_points, axis=0)
robot_mean = np.mean(robot_points, axis=0)

camera_points_centered = camera_points - camera_mean
robot_points_centered = robot_points - robot_mean

# Compute the covariance matrix
H = np.dot(camera_points_centered.T, robot_points_centered)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(H)
R_matrix = np.dot(Vt.T, U.T)

# Ensure a proper rotation matrix (det(R) should be +1)
if np.linalg.det(R_matrix) < 0:
    Vt[-1, :] *= -1
    R_matrix = np.dot(Vt.T, U.T)

# Compute the translation vector
t_vector = robot_mean - np.dot(R_matrix, camera_mean)

print("Estimated Rotation Matrix R:")
print(R_matrix)
print("Estimated Translation Vector t:")
print(t_vector)

# Construct the 4x4 homogeneous transformation matrix T
T = np.eye(4)
T[:3, :3] = R_matrix
T[:3, 3] = t_vector

print("Homogeneous Transformation Matrix T:")
print(T)

# Example camera point in homogeneous coordinates (X_c, Y_c, Z_c, 1)
example_camera_point = np.array([17.12, -20.39, 287.00, 1])

# Transform the point to the robot's base frame
transformed_point = T @ example_camera_point

# Apply sign correction if necessary
# Assuming you find x and y need to be inverted
X_r_new, Y_r_new, Z_r_new = transformed_point[0], transformed_point[1], transformed_point[2]

print(f"Transformed coordinates in robot's base frame: X={X_r_new}, Y={Y_r_new}, Z={Z_r_new}")