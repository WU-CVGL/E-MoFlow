import torch

from typing import Tuple

def slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical Linear Interpolation between quaternions"""
    cos_half_theta = torch.dot(q1, q2)
    
    if cos_half_theta < 0:
        q2 = -q2
        cos_half_theta = -cos_half_theta
    
    if cos_half_theta >= 1.0:
        return q1
    
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1 - cos_half_theta*cos_half_theta)
    
    if torch.abs(sin_half_theta) < 1e-6:
        return q1 * 0.5 + q2 * 0.5
    
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta
    
    return ratio_a * q1 + ratio_b * q2

def quaternion_to_angular_velocity(q1: torch.Tensor, q2: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Convert quaternion difference to angular velocity
    
    Args:
        q1: First quaternion [w,x,y,z]
        q2: Second quaternion [w,x,y,z]
        dt: Time difference
    
    Returns:
        omega: Angular velocity [3]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    omega = 2/dt * torch.tensor([
        w1*x2 - x1*w2 - y1*z2 + z1*y2,
        w1*y2 + x1*z2 - y1*w2 - z1*x2,
        w1*z2 - x1*y2 + y1*x2 - z1*w2
    ]).to(q1.device)
    
    return omega

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    
    q_norm = torch.norm(q)
    if abs(q_norm - 1.0) > 1e-2:
        print(f"The L2 norm of q is{q_norm}, not 1")
    
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ]).to(q.device)
    
    if torch.norm(torch.mm(R, R.t()) - torch.eye(3, device=q.device, dtype=q.dtype)) > 1e-2:
        print("Warning: Rotation matrix orthogonality check failed")
        
    if abs(torch.det(R) - 1.0) > 1e-2:
        print("Warning: Rotation matrix determinant check failed")
    return R

def angular_velocity_to_euler_rates(omega: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Convert angular velocity vector to Euler angle rates (ZYX order)
    
    Args:
        omega: Angular velocity vector [wx, wy, wz]
        q: Quaternion [w, x, y, z]
    
    Returns:
        euler_rates: Euler angle rates [roll_rate, pitch_rate, yaw_rate] 
    """
    w, x, y, z = q
    
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    transform = torch.tensor([
        [1, torch.sin(roll)*torch.tan(pitch), torch.cos(roll)*torch.tan(pitch)],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll)/torch.cos(pitch), torch.cos(roll)/torch.cos(pitch)]
    ]).to(q.device)
    
    euler_rates = transform @ omega
    
    return euler_rates

def pose_to_velocity(timestamp: float, pose: torch.Tensor, dataset_name: str=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get velocity and angular velocity at specified timestamp
    
    Args:
        timestamp: Query timestamp
        pose: Pose data tensor [N, 8] (time,x,y,z,qx,qy,qz,qw)
        dataset_name: Name of dataset
        
    Returns:
        v: Linear velocity [3]
        omega: Angular velocity [3]
    """
    # Find nearest poses
    mask_pre = pose[:,0] < timestamp
    mask_post = pose[:,0] > timestamp
    
    if not torch.any(mask_pre) or not torch.any(mask_post):
        raise ValueError("Timestamp out of bounds")
        
    time_pre = torch.where(mask_pre)[0][-1]
    time_after = torch.where(mask_post)[0][0]

    # Calculate velocity in world frame
    dt = pose[time_after,0] - pose[time_pre,0]
    v_mocap = (pose[time_after,1:4] - pose[time_pre,1:4])/dt
    
    # Interpolation parameter
    p = (timestamp - pose[time_pre,0])/dt
    
    # Get quaternions
    q1 = pose[time_pre,[7,4,5,6]]  # [w,x,y,z]
    q2 = pose[time_after,[7,4,5,6]]
    
    # Calculate angular velocity
    omega_mocap = quaternion_to_angular_velocity(q1, q2, dt)
    
    # Interpolate rotation
    q = slerp(q1, q2, p)
    R = quaternion_to_rotation_matrix(q)
    
    #================= only for Blender! =================#
    R_cam = torch.Tensor([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]]).to(q1.device) @ R

    # Transform velocities to body frame
    v_body = R_cam @ v_mocap
    omega_body = R @ omega_mocap

    # Transform to specific camera frame
    if dataset_name == 'VECtor':
        T_lcam_body = torch.tensor([
            [-0.857137023976571, 0.03276713258773897, -0.5140451703406658, 0.09127742788053987],
            [0.01322063096422759, -0.9962462506036175, -0.08554895133864114, -0.02255409664008403],
            [-0.5149187674240416, -0.08012317505073682, 0.853486344222504, -0.02986309837992267],
            [0., 0., 0., 1.]
        ])
        
        T_lcam_lev = torch.tensor([
            [0.9999407352369797, 0.009183655542749752, 0.005846920950435052, 0.0005085820608404798],
            [-0.009131364645448854, 0.9999186289230431, -0.008908070070089353, -0.04081979450823404],
            [-0.005928253827254812, 0.008854151768176144, 0.9999432282899994, -0.0140781304960408],
            [0., 0., 0., 1.]
        ])
        
        T_lev_body = torch.linalg.solve(T_lcam_lev, T_lcam_body)
        R_lev_body = T_lev_body[:3,:3]
        
        v = R_lev_body @ v_body
        omega = R_lev_body @ omega_body
        
    elif dataset_name == 'MVSEC':
        v = v_mocap
        omega = omega_mocap
        
    else:
        # print("Warning: No Coordinates Transform Found")
        v = v_body
        omega = omega_body
        
    return v, omega