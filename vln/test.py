import math

def euler_angles_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
    
    :param roll: Rotation around the X-axis in radians.
    :param pitch: Rotation around the Y-axis in radians.
    :param yaw: Rotation around the Z-axis in radians.
    :return: Quaternion as a tuple (w, x, y, z).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)

def quat_to_euler_angles(quaternion):
    """
    Convert a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    
    :param quaternion: Quaternion as a tuple (w, x, y, z).
    :return: Euler angles as a tuple (roll, pitch, yaw) in radians.
    """
    w, x, y, z = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)

# Initial Euler angles
original_euler_angles = (2.312, -1.587, -124.447)  # (roll, pitch, yaw) in radians

# Convert to quaternion
quaternion = euler_angles_to_quat(*original_euler_angles)
print("Quaternion:", quaternion)

# Convert back to Euler angles
recovered_euler_angles = quat_to_euler_angles(quaternion)
print("Recovered Euler angles:", recovered_euler_angles)

# Compare the original and recovered Euler angles
print("Difference:", [original_euler_angles[i] - recovered_euler_angles[i] for i in range(3)])
