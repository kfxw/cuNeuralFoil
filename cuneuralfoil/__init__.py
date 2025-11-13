"""
cuNeuralFoil: PyTorch/CUDA-accelerated wrappers around NeuralFoil.

"""

from .main import get_aero_from_kulfan_parameters_cuda, get_aero_from_airfoil_cuda
from .cu_kulfan_airfoil import cuKulfanAirfoil

__all__ = [
    "get_aero_from_kulfan_parameters_cuda",
    "get_aero_from_airfoil_cuda",
    "cuKulfanAirfoil",
]