import os
import enum
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


class EarlyStoppingMode(enum.Enum):
    MIN = "min"
    MAX = "max"


class EarlyStopping:
    def __init__(
        self,
        burn_in_steps: int = 0,
        mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
        min_delta: float = 0.001,
        patience: int = 150,
        percentage: bool = False,
    ):
        self.burn_in_steps = burn_in_steps
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = np.inf if mode == EarlyStoppingMode.MIN else -np.inf
        self.num_bad_epochs = 0
        self.percentage = percentage
        self.total_steps = 0

    @staticmethod
    def no_stopping():
        return EarlyStopping(patience=0)

    def _is_better(self, current_perf_metric: float) -> bool:
        if self.patience == 0:
            return True

        if not self.percentage:
            if self.mode == EarlyStoppingMode.MIN:
                return current_perf_metric < self.best - self.min_delta
            elif self.mode == EarlyStoppingMode.MAX:
                return current_perf_metric > self.best + self.min_delta
        else:
            if self.mode == EarlyStoppingMode.MIN:
                return current_perf_metric < self.best - (self.best * self.min_delta / 100)
            elif self.mode == EarlyStoppingMode.MAX:
                return current_perf_metric > self.best + (self.best * self.min_delta / 100)

    def step(self, perf_metric: float) -> bool:
        self.total_steps += 1
        if self.total_steps < self.burn_in_steps:
            return False

        if self.total_steps == self.burn_in_steps:
            self.reset()
            return False

        if self.patience == 0:
            return False

        if np.isnan(perf_metric):
            print("WARNING: NaN detected in performance metric. Early stopping.")
            return True

        if self._is_better(perf_metric):
            self.num_bad_epochs = 0
            self.best = perf_metric
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(
                f"Early stopping after {self.total_steps} steps with {self.num_bad_epochs} bad epochs (patience: {self.patience})."
            )
            return True

        return False

    def reset(self):
        self.best = np.inf if self.mode == EarlyStoppingMode.MIN else -np.inf
        self.num_bad_epochs = 0
        

@dataclass
class EarlyStoppingStats:
    """statistics for early stopping"""
    segment_iterations: List[int]       # real iterations per segment
    early_stopped_segments: List[int]   # indices of segments that were early stopped
    total_segments: int                 # total number of segments
    
    def add_segment_result(self, segment_idx: int, iterations: int, early_stopped: bool):
        """add result for a segment"""
        self.segment_iterations.append(iterations)
        if early_stopped:
            self.early_stopped_segments.append(segment_idx)
    
    def get_average_iterations(self) -> float:
        """get average iterations per segment"""
        return np.mean(self.segment_iterations) if self.segment_iterations else 0.0
    
    def get_early_stopping_rate(self) -> float:
        """rate of early stopped segments"""
        return len(self.early_stopped_segments) / self.total_segments if self.total_segments > 0 else 0.0
    
    def get_early_stopped_average_iterations(self) -> float:
        """get average iterations for early stopped segments"""
        if not self.early_stopped_segments:
            return 0.0
        early_stopped_iterations = [self.segment_iterations[i] for i in self.early_stopped_segments]
        return np.mean(early_stopped_iterations)
    
    def get_completed_average_iterations(self) -> float:
        """get average iterations for completed segments"""
        completed_segments = [i for i in range(self.total_segments) if i not in self.early_stopped_segments]
        if not completed_segments:
            return 0.0
        completed_iterations = [self.segment_iterations[i] for i in completed_segments]
        return np.mean(completed_iterations)
    
    def save_to_file(self, save_dir: str):
        """save  statistics to a file"""
        stats_file = os.path.join(save_dir, "early_stopping_stats.txt")
        
        with open(stats_file, 'w') as f:
            f.write("EARLY STOPPING STATISTICS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total segments: {self.total_segments}\n")
            f.write(f"Early stopped segments: {len(self.early_stopped_segments)} ({self.get_early_stopping_rate():.2%})\n")
            f.write(f"Completed segments: {self.total_segments - len(self.early_stopped_segments)}\n\n")
            
            f.write(f"Average iterations per segment: {self.get_average_iterations():.2f}\n")
            f.write(f"Average iterations (early stopped): {self.get_early_stopped_average_iterations():.2f}\n")
            f.write(f"Average iterations (completed): {self.get_completed_average_iterations():.2f}\n\n")
            
            f.write("Detailed results per segment:\n")
            f.write("-" * 30 + "\n")
            for i, iterations in enumerate(self.segment_iterations):
                status = "Early Stopped" if i in self.early_stopped_segments else "Completed"
                f.write(f"Segment {i:3d}: {iterations:4d} iterations ({status})\n")
            
            f.write(f"\nEarly stopped segment indices: {self.early_stopped_segments}\n")
        
        print(f"Early stopping statistics saved to: {stats_file}")