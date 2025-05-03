import torch
import kornia

def vector_dir_error_in_radians(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v1_norm = torch.norm(v1, dim=-1)
    v2_norm = torch.norm(v2, dim=-1)
    
    zero_mask = (v1_norm < eps) | (v2_norm < eps)
    
    dot_product = torch.sum(v1 * v2, dim=-1)
    cosine = dot_product / (v1_norm * v2_norm + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    
    angle = torch.where(zero_mask, torch.zeros_like(angle), angle)
    return angle

def vector_dir_error_in_degrees(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle_rad = vector_dir_error_in_radians(v1, v2, eps)
    return angle_rad * 180 / torch.pi

def vector_mag_error(v1: torch.Tensor, v2: torch.Tensor, use_abs: bool = True) -> torch.Tensor:
    
    norm1 = torch.norm(v1, dim=1)  
    norm2 = torch.norm(v2, dim=1) 
    
    diff = (norm1 - norm2) if not use_abs else torch.abs(norm1 - norm2)
    
    return diff.squeeze()

def vec2skewmat(vec):
    if len(vec.shape) == 1:
        vec = vec.unsqueeze(0)
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec).squeeze(0)
    else:
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec)
    return mat