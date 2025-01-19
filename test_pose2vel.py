import numpy as np
from scipy.spatial.transform import Rotation

def read_pose_file(filename):
    data = np.loadtxt(filename)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # [qx, qy, qz, qw]
    return timestamps, positions, quaternions

def compute_velocities_in_camera_frame(timestamps, positions, quaternions):
    n = len(timestamps)
    linear_velocities = np.zeros((n-1, 3))
    angular_velocities = np.zeros((n-1, 3))
    
    for i in range(n-1):
        dt = timestamps[i+1] - timestamps[i]
        
        # 获取世界到相机的旋转矩阵
        R_w2c = Rotation.from_quat([quaternions[i][0], quaternions[i][1], 
                                   quaternions[i][2], quaternions[i][3]]).as_matrix()
        # 定义y轴向上的相机坐标系
        R_cam = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]]) @ R_w2c
        
        dp_w = positions[i+1] - positions[i]
        linear_velocities[i] = (R_cam @ dp_w) / dt
        
        R1 = Rotation.from_quat([quaternions[i][0], quaternions[i][1], 
                                quaternions[i][2], quaternions[i][3]])
        R2 = Rotation.from_quat([quaternions[i+1][0], quaternions[i+1][1], 
                                quaternions[i+1][2], quaternions[i+1][3]])
        
        dR = R2 * R1.inv()
        angles_w = dR.as_rotvec()
        angular_velocities[i] = (R_cam @ angles_w) / dt
    
    return linear_velocities, angular_velocities


# 主程序
timestamps, positions, quaternions = read_pose_file("camera_pose.txt")
linear_vel, angular_vel = compute_velocities_in_camera_frame(timestamps, positions, quaternions)

# 打印结果
print("Camera frame linear velocities (m/s):")
print("Time\t\tvx\t\tvy\t\tvz")
for i in range(len(timestamps)-1):
    print(f"{timestamps[i]:.3f}\t{linear_vel[i][0]:.6f}\t{linear_vel[i][1]:.6f}\t{linear_vel[i][2]:.6f}")

print("\nCamera frame angular velocities (rad/s):")
print("Time\t\twx\t\twy\t\twz")
for i in range(len(timestamps)-1):
    print(f"{timestamps[i]:.3f}\t{angular_vel[i][0]:.6f}\t{angular_vel[i][1]:.6f}\t{angular_vel[i][2]:.6f}")