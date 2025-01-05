import torch
import kornia
import theseus as th

from typing import Any, List, Optional

def vector_to_skew(vec):
    if len(vec.shape) == 1:
        vec = vec.unsqueeze(0)
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec).squeeze(0)
    else:
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec)
    return mat

class Pixel2Cam:
    def __init__(self, H: int, W: int, device: torch.device):
        self.grid = kornia.utils.create_meshgrid(
            H, W, normalized_coordinates=False).to(device)
        self.grid = self.grid.squeeze(0)
        self.ones = torch.ones(H, W, 1, device=device)
        self.pixels_homogeneous = torch.cat(
            [self.grid, self.ones], dim=-1)  # [H,W,3]
        
    def __call__(self, K: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if K.dim() == 2:
            K = K.unsqueeze(0)
        B = K.shape[0]
        
        pixels_batch = self.pixels_homogeneous.unsqueeze(0).expand(B, -1, -1, -1).to(K.device)
        
        K_4x4 = torch.zeros(B, 4, 4, device=K.device)
        K_4x4[:, :3, :3] = K
        K_4x4[:, 3, 3] = 1.0
        K_4x4_inv = torch.inverse(K_4x4)
        
        if depth is None:
            depth = torch.ones(B, 1, *self.grid.shape[:2], device=K.device)
            
        return kornia.geometry.camera.pixel2cam(
            depth, K_4x4_inv, pixels_batch)[..., :3]

class UnitVectorManifold(th.Manifold):
    """Unit vector manifold (S²) for normalized linear velocity"""
    
    @staticmethod
    def _init_tensor(*args: Any) -> torch.Tensor:
        # 初始化为单位向量
        v = torch.randn(1, 3)
        return v / torch.norm(v, p=2, dim=-1, keepdim=True)

    def dof(self) -> int:
        return 2  # S² 的自由度为2（3维空间中的2维球面）

    def _local_impl(
        self, variable2: "UnitVectorManifold", jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        v1 = self.tensor
        v2 = variable2.tensor
        
        # 计算球面上的切空间差异
        dot_product = torch.sum(v1 * v2, dim=-1, keepdim=True)
        # 确保数值稳定性
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        # 计算测地线距离（大圆距离）
        theta = torch.acos(dot_product)
        # 计算切空间中的向量
        normal = v2 - dot_product * v1
        normal = normal / (torch.norm(normal, p=2, dim=-1, keepdim=True) + 1e-7)
        delta = theta * normal
        
        if jacobians is not None:
            # 计算雅可比矩阵
            if len(jacobians) > 0:
                J1 = -torch.eye(3, device=v1.device).unsqueeze(0)
                jacobians[0] = J1
            if len(jacobians) > 1:
                J2 = torch.eye(3, device=v1.device).unsqueeze(0)
                jacobians[1] = J2
        
        return delta

    def _retract_impl(self, delta: torch.Tensor) -> "UnitVectorManifold":
        v = self.tensor
        # 计算退化映射（将切空间向量映射回流形）
        delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
        cos_theta = torch.cos(delta_norm)
        sin_theta = torch.sin(delta_norm)
        
        # 使用Rodrigues旋转公式
        v_new = cos_theta * v + sin_theta * delta / (delta_norm + 1e-7)
        
        return UnitVectorManifold(tensor=v_new)

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        # 将梯度投影到切空间
        v = self.tensor
        v_dot_grad = torch.sum(v * euclidean_grad, dim=-1, keepdim=True)
        return euclidean_grad - v_dot_grad * v

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / torch.norm(tensor, p=2, dim=-1, keepdim=True)

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            return False
        # 检查是否为单位向量
        norms = torch.norm(tensor, p=2, dim=1)
        return torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def _copy_impl(self, new_name: Optional[str] = None) -> "UnitVectorManifold":
        return UnitVectorManifold(
            tensor=self.tensor.clone(),
            name=new_name,
        )

class S2(th.Manifold):
    """S2流形类，用于表示单位球面上的点"""
    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None, 
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict_checks: bool = False,
        disable_checks: bool = False,
    ):
        super().__init__(
            tensor=tensor,
            name=name,
            dtype=dtype,
            strict_checks=strict_checks,
            disable_checks=disable_checks,
        )

    @staticmethod
    def _init_tensor() -> torch.Tensor:
        # 初始化为[0,0,1]方向
        return torch.tensor([0.0, 0.0, 1.0]).view(1, 3)

    def dof(self) -> int:
        return 2  # S2的自由度为2

    @staticmethod
    def rand(*size: int, generator: Optional[torch.Generator] = None,
            dtype: Optional[torch.dtype] = None, 
            device: Optional[str] = None,
            requires_grad: bool = False) -> "S2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        # 随机生成单位球面上的点
        v = torch.randn(size[0], 3, generator=generator, 
                       dtype=dtype, device=device, 
                       requires_grad=requires_grad)
        return S2(tensor=v / torch.norm(v, p=2, dim=1, keepdim=True))

    @staticmethod 
    def randn(*size: int, generator: Optional[torch.Generator] = None,
             dtype: Optional[torch.dtype] = None,
             device: Optional[str] = None, 
             requires_grad: bool = False) -> "S2":
        return S2.rand(*size, generator=generator, dtype=dtype,
                      device=device, requires_grad=requires_grad)

    def _local_impl(
        self, variable2: "S2", jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """计算两点间的测地线，返回起点切空间中的切向量
        
        Args:
            variable2: 终点 (在S²上的点)
            jacobians: 如果提供，计算对起点和终点的雅可比矩阵
            
        Returns:
            delta: 切空间中的向量，表示从起点到终点的测地线方向和长度
        """
        v1 = self.tensor  # 起点p, shape: (B,3)
        v2 = variable2.tensor  # 终点q, shape: (B,3)
        
        # 计算夹角
        dot_product = torch.sum(v1 * v2, dim=1, keepdim=True)  # shape: (B,1)
        # 确保数值稳定性，避免反三角函数的导数奇异
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot_product)  # shape: (B,1)
        
        # 特殊情况处理：如果两点重合
        identical_points = theta < 1e-10
        
        # 计算切空间方向 (从v1到v2的方向在v1切空间中的投影)
        normal = v2 - dot_product * v1  # shape: (B,3)
        normal_norm = torch.norm(normal, p=2, dim=1, keepdim=True)  # shape: (B,1)
        
        # 处理重合点的情况
        safe_norm = torch.where(identical_points, 
                            torch.ones_like(normal_norm),
                            normal_norm)
        normal = normal / safe_norm
        
        # 生成切空间中的向量表示 (方向 * 长度)
        delta = theta * normal  # shape: (B,3)

        if jacobians is not None:
            # 计算雅可比矩阵
            small_angle_threshold = 1e-4
            small_angle = theta < small_angle_threshold
            
            # 预计算一些常用值
            sin_theta = torch.sin(theta)  # shape: (B,1)
            I3 = torch.eye(3, device=v1.device).unsqueeze(0)  # shape: (1,3,3)
            
            if len(jacobians) > 0:
                # 计算关于起点的雅可比 J1
                v1_out = v1.unsqueeze(2)  # shape: (B,3,1)
                v1_in = v1.unsqueeze(1)   # shape: (B,1,3)
                proj_p = I3 - v1_out @ v1_in  # shape: (B,3,3)
                
                # 对小角度使用Taylor展开近似
                J1_small = -proj_p * (1.0 + theta.unsqueeze(-1) * theta.unsqueeze(-1) / 6.0)
                
                # 对大角度使用完整公式
                safe_sin = torch.where(small_angle, 
                                    torch.ones_like(sin_theta),
                                    sin_theta)
                J1_large = -proj_p / safe_sin.unsqueeze(-1)
                
                # 根据角度选择适当的计算结果
                J1 = torch.where(small_angle.unsqueeze(-1).unsqueeze(-1),
                            J1_small, J1_large)
                
                # 处理重合点的情况
                J1 = torch.where(identical_points.unsqueeze(-1).unsqueeze(-1),
                            -I3.expand(v1.shape[0], -1, -1),
                            J1)
                
                jacobians[0] = J1
                
            if len(jacobians) > 1:
                # 计算关于终点的雅可比 J2
                v2_out = v2.unsqueeze(2)  # shape: (B,3,1)
                v2_in = v2.unsqueeze(1)   # shape: (B,1,3)
                proj_q = I3 - v2_out @ v2_in  # shape: (B,3,3)
                
                # 对小角度使用Taylor展开近似
                J2_small = proj_q * (1.0 + theta.unsqueeze(-1) * theta.unsqueeze(-1) / 6.0)
                
                # 对大角度使用完整公式
                J2_large = proj_q / safe_sin.unsqueeze(-1)
                
                # 根据角度选择适当的计算结果
                J2 = torch.where(small_angle.unsqueeze(-1).unsqueeze(-1),
                            J2_small, J2_large)
                
                # 处理重合点的情况
                J2 = torch.where(identical_points.unsqueeze(-1).unsqueeze(-1),
                            I3.expand(v1.shape[0], -1, -1),
                            J2)
                
                jacobians[1] = J2

        return delta

    def _retract_impl(self, delta: torch.Tensor) -> "S2":
        """实现从切空间到流形的指数映射，包含完整的特殊情况处理
        
        Args:
            delta: 切空间中的向量 (B,3)
            
        Returns:
            新的S²点
        """
        v = self.tensor  # 当前点 (B,3)
        
        # 1. 计算切空间向量的范数
        delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)  # (B,1)
        
        # 2. 处理特殊情况
        # 2.1 极小位移
        tiny_step = delta_norm < 1e-10
        # 2.2 接近π的位移（几乎antipodal points）
        large_step = delta_norm > math.pi - 1e-10
        # 2.3 数值不稳定区域
        unstable_zone = delta_norm < 1e-4
        
        # 3. 安全的范数计算
        safe_delta_norm = torch.where(tiny_step, 
                                    torch.ones_like(delta_norm),
                                    delta_norm)
        
        # 4. 计算旋转轴方向
        u = delta / safe_delta_norm  # (B,3)
        
        # 5. 计算三角函数值
        cos_theta = torch.cos(delta_norm)  
        sin_theta = torch.sin(delta_norm)
        
        # 6. 在不稳定区域使用泰勒展开近似
        # cos(θ) ≈ 1 - θ²/2
        # sin(θ) ≈ θ - θ³/6
        cos_theta_small = 1.0 - delta_norm * delta_norm / 2.0
        sin_theta_small = delta_norm - delta_norm * delta_norm * delta_norm / 6.0
        
        cos_theta = torch.where(unstable_zone, cos_theta_small, cos_theta)
        sin_theta = torch.where(unstable_zone, sin_theta_small, sin_theta)
        
        # 7. 计算结果
        # 7.1 常规情况：使用Rodrigues公式
        result = (v * cos_theta + 
                u * sin_theta + 
                torch.cross(u, v) * (1.0 - cos_theta))
        
        # 7.2 处理极小位移：返回原点
        result = torch.where(tiny_step.expand_as(result), 
                            v, 
                            result)
        
        # 7.3 处理接近π的位移：使用稳定的计算方式
        antipodal_result = -v + 2.0 * (torch.sum(v * u, dim=1, keepdim=True) * u)
        result = torch.where(large_step.expand_as(result),
                            antipodal_result,
                            result)
        
        # 8. 确保数值精度：重新归一化
        result = result / torch.norm(result, p=2, dim=1, keepdim=True)
        
        return S2(tensor=result)

    def _project_impl(self,
                    euclidean_grad: torch.Tensor,
                    is_sparse: bool = False) -> torch.Tensor:
        """将欧氏空间的梯度投影到S2流形的切空间
        
        Args:
            euclidean_grad: 欧氏空间中的梯度 (B,3)
            is_sparse: 是否为稀疏梯度，默认False
            
        Returns:
            切空间中的梯度 (B,3)
        """
        x = self.tensor  # 当前点 (B,3)
        
        # 计算内积 <x,grad>
        inner_prod = torch.sum(x * euclidean_grad, dim=1, keepdim=True)  # (B,1)
        
        # 投影公式: grad - <x,grad>x
        proj_grad = euclidean_grad - inner_prod * x
        
        return proj_grad

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / torch.norm(tensor, p=2, dim=1, keepdim=True)

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            return False
        norms = torch.norm(tensor, p=2, dim=1)
        return torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def _copy_impl(self, new_name: Optional[str] = None) -> "S2":
        return S2(
            tensor=self.tensor.clone(),
            name=new_name,
        )

class PoseOptimizer:
    def __init__(self, image_size, device="cuda"):
        self.H, self.W = image_size
        self.device = device
        
    def create_cost_function(self, normalized_coords, optical_flow):
        
        v = th.Vector(dof=3, name="linear_velocity") 
        w = th.Vector(dof=3, name="angular_velocity")
        
        x = th.Variable(normalized_coords, name="norm_coords")  # [1,N,3]
        u = th.Variable(optical_flow, name="optical_flow")      # [1,N,3]
        
        def dec_error_fn(optim_vars, aux_vars):
            v_vec, w_vec = optim_vars
            x_batch, u_batch = aux_vars
            v_vec_tensor, w_vec_tensor = v_vec.tensor, w_vec.tensor
            x_batch_tensor, u_batch_tensor = x_batch.tensor, u_batch.tensor
            
            x_batch_tensor = x_batch_tensor.squeeze(0)
            u_batch_tensor = u_batch_tensor.squeeze(0)
            v_skew = vector_to_skew(v_vec_tensor).squeeze(0)
            w_skew = vector_to_skew(w_vec_tensor).squeeze(0)
            
            s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew))
            
            # calculate v_skew @ x & s @ x
            v_skew_x = torch.matmul(v_skew, x_batch_tensor.transpose(0,1)).transpose(0,1) # [N,3]
            s_x = torch.matmul(s, x_batch_tensor.transpose(0,1)).transpose(0,1)  # [N,3]
            
            term1 = torch.sum(u_batch_tensor * v_skew_x, dim=1) # [N]
            term2 = torch.sum(x_batch_tensor * s_x, dim=1)  # [N]
            error = term1 - term2
            
            return error.unsqueeze(0)

        optim_vars = [v, w]
        aux_vars = [x, u]
        dec_cost_fn = th.AutoDiffCostFunction(
            optim_vars,
            dec_error_fn,
            x.shape[1],
            aux_vars=aux_vars,
            name="dec_cost_fn"
        )
        
        return dec_cost_fn
    
    def optimize(self, normalized_coords, optical_flow, num_iterations=1000):
        # check
        assert normalized_coords.shape[-1] == 3, "[ERROR] Normalized coordinates should be in homogeneous form [x,y,1]"
        assert optical_flow.shape[-1] == 3, "[ERROR] Optical flow should be in homogeneous form [u,v,0]"
        assert normalized_coords.shape[:-1] == optical_flow.shape[:-1], "[ERROR] Batch dimensions should match"
        
        normalized_coords = normalized_coords.unsqueeze(0)
        optical_flow = optical_flow.unsqueeze(0)
        
        objective = th.Objective()
        cost_functions = self.create_cost_function(
            normalized_coords, optical_flow
        )
        objective.add(cost_functions)
            
        optimizer = th.GaussNewton(
            objective,
            max_iterations=num_iterations,
            step_size=0.1,
        )

        v_init = torch.zeros((1,3), device=self.device)
        w_init = torch.zeros((1,3), device=self.device)
        theseus_inputs = {
            "linear_velocity": v_init,
            "angular_velocity": w_init,
            "norm_coords": normalized_coords,
            "optical_flow": optical_flow
        }
        
        theseus_optim = th.TheseusLayer(optimizer).to(self.device)
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                },
            )
        
        v_opt = info.best_solution["linear_velocity"] 
        w_opt = info.best_solution["angular_velocity"]
        
        return v_opt, w_opt