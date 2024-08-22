

import numpy as np

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R

def compute_camera_view_transform(quaternion, position):
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.array(position)
    
    # Compute the view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R.T  # Transpose of rotation matrix
    view_matrix[:3, 3] = -R.T @ T  # Apply the inverse translation
    
    return view_matrix

# Example quaternion and position
quaternion = [0.14632428, 0.21607629, 0.02670759, 0.96498028]  # Example values
position = [5.9468007, -28.225435, 1.3916627]  # Example values

camera_view_transform = compute_camera_view_transform(quaternion, position)
print(camera_view_transform)
