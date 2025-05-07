import torch
import time

class TimeAnalyzer:
    def __init__(self):
        self.epoch_start_time = None
        self.total_train_time = 0
        self.total_valid_time = 0        
        self.epoch_times = []
        self.valid_times = []
        
    def start_epoch(self):
        """Start timing an epoch"""
        torch.cuda.synchronize()  
        self.epoch_start_time = time.perf_counter()
        
    def end_epoch(self):
        """End timing an epoch"""
        torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.total_train_time += epoch_time
        
    def start_valid(self):
        """Start timing valid phase"""
        torch.cuda.synchronize()
        self.valid_start_time = time.perf_counter()
        
    def end_valid(self):
        """End timing valid phase"""
        torch.cuda.synchronize()
        valid_end_time = time.perf_counter()
        valid_time = valid_end_time - self.valid_start_time
        self.valid_times.append(valid_time)
        self.total_valid_time += valid_time
        
    def get_statistics(self):
        """Get timing statistics"""
        avg_epoch_time = self.total_train_time / len(self.epoch_times)
        avg_valid_time = self.total_valid_time / len(self.valid_times)
        return {
            'total_train_time': self.total_train_time / 60.0,
            'avg_train_time': avg_epoch_time / 60.0,
            'total_valid_time': self.total_valid_time / 60.0,
            'avg_valid_time': avg_valid_time / 60.0
        }