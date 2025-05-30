from torch.utils.data import Dataset
import torch
import numpy as np

class WindowSampler(Dataset):
    def __init__(self, data, window_size=32, overlapping=False, shuffle_windows=True):
        """
        Args:
            data: Tensor of shape [num_days, d_feat]
            window_size: Number of consecutive rows (T).
            overlapping: If True, use sliding (overlapping) windows.
            shuffle_windows: Whether to shuffle the windows.
        """
        self.data = data
        self.window_size = window_size
        self.overlapping = overlapping

        # Generate start indices for windowing
        if overlapping:
            self.indices = list(range(len(data) - window_size + 1))  # Sliding windows
        else:
            self.indices = list(range(0, len(data), window_size))  # Non-overlapping

        if shuffle_windows:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """ Return one window of shape `[≤T, d_feat]`, allowing shorter windows at the end. """
        start_idx = self.indices[idx]
        window = self.data[start_idx : start_idx + self.window_size]  # May be shorter than T

        return window.clone().detach().float()  # Shape: [≤T, d_feat]
