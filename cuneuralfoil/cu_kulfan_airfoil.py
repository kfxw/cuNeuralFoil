import numpy as np
import torch
from aerosandbox import KulfanAirfoil
import copy as _copy

class cuKulfanAirfoil(KulfanAirfoil):
    """
    A wrapper subclass around KulfanAirfoil that is constructed from an
    existing KulfanAirfoil instance.

    Usage:
        base = KulfanAirfoil(...)  # however you normally build it

        # use the same instance, just with extra methods from cuKulfanAirfoil
        wrapped = cuKulfanAirfoil(base)

        # or make a deep copy first
        wrapped_copy = cuKulfanAirfoil(base, copy_instance=True)
    """

    def __new__(cls, base: KulfanAirfoil, *, copy_instance: bool = False, **kargs):
        obj = _copy.deepcopy(base) if copy_instance else base
        obj.__class__ = cls
        return obj

    def __init__(self, base: KulfanAirfoil, *, copy_instance: bool = False, requires_grad: bool = False, device: torch.device = None):
        # Do NOT call super().__init__ here
        self._from_copy = copy_instance
        
        # setup cuda tensor environment
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.requires_grad = requires_grad
        
        # Handle the Kulfan parameters
        # equivalent to: 
        #    self.lower_weights_cuda = torch.as_tensor(self.lower_weights, dtype=torch.float32, device=self.device)
        #    self.lower_weights_cuda.requires_grad = requires_grad
        #    del self.lower_weights
        def _init_param(name: str):
            raw = self.__dict__[name]
            raw_np = np.array(raw, dtype=np.float32)
            t = torch.tensor(
                raw_np,
                dtype=torch.float32,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            setattr(self, f"{name}_cuda", t)

            # clear the old raw attribute so property fully takes over
            del self.__dict__[name]

        for pname in [
            "lower_weights",
            "upper_weights",
            "leading_edge_weight",
            "TE_thickness",
            "N1",
            "N2",
        ]:
            _init_param(pname)
            
            
    @property
    def kulfan_parameters_cuda(self):
        return {
            "lower_weights_cuda": self.lower_weights_cuda,
            "upper_weights_cuda": self.upper_weights_cuda,
            "leading_edge_weight_cuda": self.leading_edge_weight_cuda,
            "TE_thickness_cuda": self.TE_thickness_cuda,
        }
    
    # ---- Properties to auto-sync CPU <-> CUDA tensor, making it compatible with neuralfoil operations ----
    @property
    def lower_weights(self) -> np.ndarray:
        return self.lower_weights_cuda.detach().cpu().numpy()

    @lower_weights.setter
    def lower_weights(self, value):
        with torch.no_grad():
            self.lower_weights_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.lower_weights_cuda.requires_grad_(self.requires_grad)

    @property
    def upper_weights(self) -> np.ndarray:
        return self.upper_weights_cuda.detach().cpu().numpy()

    @upper_weights.setter
    def upper_weights(self, value):
        with torch.no_grad():
            self.upper_weights_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.upper_weights_cuda.requires_grad_(self.requires_grad)

    @property
    def leading_edge_weight(self) -> float:
        return float(self.leading_edge_weight_cuda.detach().cpu().item())

    @leading_edge_weight.setter
    def leading_edge_weight(self, value):
        with torch.no_grad():
            self.leading_edge_weight_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.leading_edge_weight_cuda.requires_grad_(self.requires_grad)

    @property
    def TE_thickness(self) -> float:
        return float(self.TE_thickness_cuda.detach().cpu().item())

    @TE_thickness.setter
    def TE_thickness(self, value):
        with torch.no_grad():
            self.TE_thickness_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.TE_thickness_cuda.requires_grad_(self.requires_grad)

    @property
    def N1(self) -> float:
        return float(self.N1_cuda.detach().cpu().item())

    @N1.setter
    def N1(self, value):
        with torch.no_grad():
            self.N1_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.N1_cuda.requires_grad_(self.requires_grad)

    @property
    def N2(self) -> float:
        return float(self.N2_cuda.detach().cpu().item())

    @N2.setter
    def N2(self, value):
        with torch.no_grad():
            self.N2_cuda = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.N2_cuda.requires_grad_(self.requires_grad)
            