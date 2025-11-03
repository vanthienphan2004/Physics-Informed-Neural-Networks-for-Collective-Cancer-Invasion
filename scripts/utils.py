"""
Utility functions for the Cancer Invasion Simulation.

This module contains helper functions for device detection.
"""

import torch


def get_compute_device() -> torch.device:
    """
    Detect and return the appropriate PyTorch device for computation.

    Prioritizes CUDA, then MPS (Apple Silicon), and finally CPU.

    Returns
    -------
    torch.device
        The detected compute device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device