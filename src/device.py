"""Выбор устройства для PyTorch: NVIDIA GPU (CUDA) → Apple GPU (MPS) → CPU."""
import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
