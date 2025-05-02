import math
from torch.optim.lr_scheduler import LambdaLR

def create_expcos_scheduler(optimizer, lr_start, lr_end_exp, lr_end_cos, exp_steps, cos_steps):
    assert exp_steps > 0, "exp_steps must be greater than 0"
    assert cos_steps > 0, "cos_steps must be greater than 0"
    assert lr_start > lr_end_exp > lr_end_cos, "Learning rates must satisfy lr_start > lr_end_exp > lr_end_cos"
    
    def lr_lambda(current_step):
        # exp decay
        if current_step < exp_steps:
            decay_factor = (lr_end_exp / lr_start) ** (current_step / exp_steps)
            return decay_factor
        
        # cosine decay
        t = current_step - exp_steps
        if t >= cos_steps:
            return lr_end_cos / lr_start
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / cos_steps))
        current_lr = lr_end_cos + (lr_end_exp - lr_end_cos) * cosine_factor
        return current_lr / lr_start
    
    # initiate optimizer
    optimizer.param_groups[0]['lr'] = lr_start
    return LambdaLR(optimizer, lr_lambda)

def create_cosexp_scheduler(optimizer, lr_start, lr_end_cos, lr_end_exp, cos_steps, exp_steps):
    assert cos_steps > 0 and exp_steps > 0, "cos_steps must be greater than 0"
    assert lr_start > lr_end_cos,  "Learning rates must satisfy lr_start > lr_end_cos > lr_end_exp"
    
    def lr_lambda(current_step):
        if current_step < cos_steps:
            t = current_step
            cosine_factor = 0.5 * (1 + math.cos(math.pi * t / cos_steps))
            current_lr = lr_end_cos + (lr_start - lr_end_cos) * cosine_factor
            return current_lr / lr_start
        
        t = current_step - cos_steps
        if t >= exp_steps:
            return lr_end_exp / lr_start
        
        decay_factor = (lr_end_exp / lr_end_cos) ** (t / exp_steps)
        current_lr = lr_end_cos * decay_factor
        return current_lr / lr_start
    
    # initiate optimizer
    optimizer.param_groups[0]['lr'] = lr_start
    return LambdaLR(optimizer, lr_lambda)

def create_exponential_scheduler(optimizer, lr1, lr2, total_steps):
    assert lr1 > lr2 > 0, "Learning rates must satisfy lr_1 > lr_2 > 0"
    assert total_steps > 0, "total_steps must be greater than 0"

    def lr_lambda(current_step):
        if current_step >= total_steps:
            return lr2 / lr1  
        decay_factor = (lr2 / lr1) ** (current_step / total_steps)
        return decay_factor

    optimizer.param_groups[0]['lr'] = lr1
    return LambdaLR(optimizer, lr_lambda)

def create_cosine_scheduler(optimizer, lr1, lr2, total_steps):
    assert lr1 > lr2 > 0, "lr1 must be greater than lr2 and both must be positive"
    assert total_steps > 0, "total_step must be a positive integer"

    def lr_lambda(current_step):
        if current_step >= total_steps:
            return lr2 / lr1 
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
        current_lr = lr2 + (lr1 - lr2) * cosine_decay
        return current_lr / lr1 

    optimizer.param_groups[0]['lr'] = lr1
    return LambdaLR(optimizer, lr_lambda)