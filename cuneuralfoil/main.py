import torch
from cuneuralfoil.cu_kulfan_airfoil import cuKulfanAirfoil
import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil
from typing import Union, Dict, Set, List, Iterable
from pathlib import Path
import re
from neuralfoil._basic_data_type import Data

nn_weights_dir = Path(neuralfoil.__file__).parent / "nn_weights_and_biases"

# Here, we compute a small epsilon value, which is used later to clip values to suppress overflow.
# This looks a bit complicated below, but it's basically just a dynamic way to avoid explicitly referring to float bit-widths.
_eps: torch.Tensor = torch.tensor(10 / np.finfo(np.array(1.0).dtype).max)
_ln_eps: torch.Tensor = torch.log(_eps)
    
def _sigmoid(x: torch.Tensor) -> Union[float, torch.Tensor]:
    global _ln_eps
    _ln_eps = _ln_eps.to(x.device)
    x = torch.clamp(x, _ln_eps, -_ln_eps)  # Clip to suppress overflow
    return 1 / (1 + torch.exp(-x))

### For speed, pre-loads parameters with statistics about the training distribution
# Includes the mean, covariance, and inverse covariance of training data in the input latent space (25-dim)
_scaled_input_distribution = dict(
    np.load(nn_weights_dir / "scaled_input_distribution.npz")
)
_scaled_input_distribution["N_inputs"]: int = len(
    _scaled_input_distribution["mean_inputs_scaled"]
)
_scaled_input_distribution["mean_inputs_scaled"] = torch.as_tensor(
    _scaled_input_distribution["mean_inputs_scaled"],
    dtype=torch.float32,
)
_scaled_input_distribution["inv_cov_inputs_scaled"] = torch.as_tensor(
    _scaled_input_distribution["inv_cov_inputs_scaled"],
    dtype=torch.float32,
)

### For speed, pre-loads the neural network weights and biases
_nn_parameter_files: Iterable[Path] = nn_weights_dir.glob("nn-*.npz")
_allowable_model_sizes: set[str] = set(
    [
        # regex parse, which results in the strings "large", "medium", "small", etc.
        re.search(r"nn-(.*).npz", str(path)).group(1)
        for path in _nn_parameter_files
    ]
)
_nn_parameters: dict[str, dict[str, np.ndarray]] = {
    model_size: dict(np.load(nn_weights_dir / f"nn-{model_size}.npz"))
    for model_size in _allowable_model_sizes
}
    
def _squared_mahalanobis_distance_cuda(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Mahalanobis distance of a set of points from the training data.

    Args:
        x: Query point in the input latent space. Shape: (N_cases, N_inputs)
            For non-vectorized queries, N_cases=1.

    Returns:
        The squared Mahalanobis distance. Shape: (N_cases,)
    """
    global _scaled_input_distribution
    _scaled_input_distribution["mean_inputs_scaled"] = _scaled_input_distribution["mean_inputs_scaled"].to(x.device)
    _scaled_input_distribution["inv_cov_inputs_scaled"] = _scaled_input_distribution["inv_cov_inputs_scaled"].to(x.device)
    mean = _scaled_input_distribution["mean_inputs_scaled"].view(1, -1)
    inv_cov = _scaled_input_distribution["inv_cov_inputs_scaled"]
    x_minus_mean = x - mean
    return torch.sum(x_minus_mean @ inv_cov * x_minus_mean, dim=1)

def get_aero_from_kulfan_parameters_cuda(
    kulfan_parameters_cuda: dict[str, torch.Tensor],
    alpha: torch.Tensor,
    Re: torch.Tensor,
    n_crit: torch.Tensor = torch.tensor(9.0),
    xtr_upper: torch.Tensor = torch.tensor(1.0),
    xtr_lower: torch.Tensor = torch.tensor(1.0),
    model_size: str = "xlarge",
    device: Union[torch.device, str] = "cpu"
) -> dict[str, torch.Tensor]:
    """
    Torch/CUDA version of NeuralFoil's aero surrogate.

    Supports both single and batched evaluation:
      - upper/lower weights: (8,) or (B, 8)
      - alpha, Re, n_crit, xtr_upper, xtr_lower: scalar or (B,)
    """

    ### Validate inputs
    if model_size not in _allowable_model_sizes:
        raise ValueError(
            f"Invalid {model_size=}. Must be one of {_allowable_model_sizes}."
        )
    nn_params_np: dict[str, np.ndarray] = _nn_parameters[model_size]
        
    ### setup cuda environment
    kulfan_parameters_cuda = {k: v.to(device) for k, v in kulfan_parameters_cuda.items()}
    alpha = alpha.to(device)
    Re = Re.to(device)
    n_crit = n_crit.to(device)
    xtr_upper = xtr_upper.to(device)
    xtr_lower = xtr_lower.to(device)
    nn_params_cuda: dict[str, torch.Tensor] = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device) 
        for k, v in nn_params_np.items()
    }
    ### Prepare the inputs for the neural network (batched)
    # Expected shapes:
    #   upper_weights_cuda: (B, 8) or (8,)
    #   lower_weights_cuda: (B, 8) or (8,)
    #   leading_edge_weight_cuda, TE_thickness_cuda: (B,) or scalar
    #   alpha, Re, n_crit, xtr_upper, xtr_lower: (B,) or scalar
    def ensure_2d_8(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device).float()
        if t.ndim == 1:          # (8,) -> (1, 8)
            t = t.unsqueeze(0)
        elif t.ndim != 2:
            raise ValueError(f"Expected 1D or 2D tensor for weights, got shape {t.shape}")
        if t.shape[1] != 8:
            raise ValueError(f"Expected 8 Kulfan weights, got shape {t.shape}")
        return t
    
    def ensure_1d_batch(t: torch.Tensor, B: int) -> torch.Tensor:
        t = t.to(device).float()
        if t.ndim == 0:
            return t.expand(B)    # scalar → broadcast to (B,)
        if t.ndim == 1 and t.shape[0] == B:
            return t
        raise ValueError(f"Expected shape ({B},) or scalar, got {t.shape}")
        
    # Kulfan weights
    upper = ensure_2d_8(kulfan_parameters_cuda["upper_weights_cuda"])
    lower = ensure_2d_8(kulfan_parameters_cuda["lower_weights_cuda"])
    B = upper.shape[0]  # number of cases in the batch
    
    # Scalar-ish Kulfan params
    LE = ensure_1d_batch(kulfan_parameters_cuda["leading_edge_weight_cuda"], B)
    TE = ensure_1d_batch(kulfan_parameters_cuda["TE_thickness_cuda"], B)
    
    # Flow parameters (alpha, Re, etc.)
    alpha = ensure_1d_batch(alpha, B)
    Re = ensure_1d_batch(Re, B)
    n_crit = ensure_1d_batch(n_crit, B)
    xtr_upper = ensure_1d_batch(xtr_upper, B)
    xtr_lower = ensure_1d_batch(xtr_lower, B)
    
    # Build features
    deg2rad = torch.pi / 180.0
    sin2a = torch.sin(2.0 * alpha * deg2rad)          # np.sind(2 * alpha)
    cos_a = torch.cos(alpha * deg2rad)              # np.cosd(alpha)
    Re_scaled = (torch.log(Re) - 12.5) / 3.5
    ncrit_scaled = (n_crit - 9.0) / 4.5
    
    # x shape: (B, N_inputs)
    x = torch.cat([
            upper,                         # (B, 8)
            lower,                         # (B, 8)
            LE.unsqueeze(-1),                 # (B, 1)
            (TE * 50.0).unsqueeze(-1),            # (B, 1)
            sin2a.unsqueeze(-1),               # (B, 1)
            cos_a.unsqueeze(-1),               # (B, 1)
            (1.0 - cos_a ** 2).unsqueeze(-1),       # (B, 1), 1 - cos^2(alpha)
            Re_scaled.unsqueeze(-1),            # (B, 1)
            ncrit_scaled.unsqueeze(-1),          # (B, 1)
            xtr_upper.unsqueeze(-1),            # (B, 1)
            xtr_lower.unsqueeze(-1),            # (B, 1)
        ], dim=1)

    ##### Evaluate the neural network

    ### First, determine what the structure of the neural network is (i.e., how many layers it has) by looking at the keys.
    # These keys come from the dictionary of saved weights/biases for the specified neural network.
    try:
        layer_indices: Set[int] = set(
            [int(key.split(".")[1]) for key in nn_params_cuda.keys()]
        )
    except TypeError:
        raise ValueError(
            f"Got an unexpected neural network file format.\n"
            f"Dictionary keys should be strings of the form 'net.0.weight', 'net.0.bias', 'net.2.weight', etc.'.\n"
            f"Instead, got keys of the form {nn_params_cuda.keys()}.\n"
        )
    layer_indices: List[int] = sorted(list(layer_indices))

    ### Neural net forward in torch
    def net(x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the raw network (taking in scaled inputs and returning scaled outputs).

        Works in the input and output latent spaces.

        Input `x` shape: (N_cases, N_inputs)
        Output `y` shape: (N_cases, N_outputs)
        """
        x = x.T
        layer_indices_to_iterate = layer_indices.copy()

        while len(layer_indices_to_iterate) != 0:
            i = layer_indices_to_iterate.pop(0)
            w = nn_params_cuda[f"net.{i}.weight"]
            b = nn_params_cuda[f"net.{i}.bias"]
            x = w @ x + b.view(-1, 1)

            if (len(layer_indices_to_iterate) != 0):  
                # swish(x) = x * sigmoid(x)
                x = x * _sigmoid(x)
        return x.T

    y = net(x)
    y[:, 0] = y[:, 0] - _squared_mahalanobis_distance_cuda(x) / (
        2 * _scaled_input_distribution["N_inputs"]
    )
    # This was baked into training in order to ensure the network asymptotes to zero analysis confidence far away from the training data.

    ### Then, flip the inputs and evaluate the network again.
    # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.
    # (This was also performed during training, so the network is "intended" to be evaluated this way.)

    x_flipped = x.clone()
    x_flipped[:, :8] = (
        x[:, 8:16] * -1
    )  # switch kulfan_lower with a flipped kulfan_upper
    x_flipped[:, 8:16] = (
        x[:, :8] * -1
    )  # switch kulfan_upper with a flipped kulfan_lower
    x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
    x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
    x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
    x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

    y_flipped = net(x_flipped)
    y_flipped[:, 0] = y_flipped[:, 0] - _squared_mahalanobis_distance_cuda(x_flipped) / (
        2 * _scaled_input_distribution["N_inputs"]
    )
    # This was baked into training in order to ensure the network asymptotes to zero analysis confidence far away from the training data.

    ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
    y_unflipped = y_flipped.clone()
    y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
    y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
    y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
    y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

    # switch upper and lower Ret, H
    y_unflipped[:, 6 : 6 + 32 * 2] = y_flipped[:, 6 + 32 * 3 : 6 + 32 * 5]
    y_unflipped[:, 6 + 32 * 3 : 6 + 32 * 5] = y_flipped[:, 6 : 6 + 32 * 2]

    # switch upper_bl_ue/vinf with lower_bl_ue/vinf
    y_unflipped[:, 6 + 32 * 2 : 6 + 32 * 3] = -1 * y_flipped[:, 6 + 32 * 5 : 6 + 32 * 6]
    y_unflipped[:, 6 + 32 * 5 : 6 + 32 * 6] = -1 * y_flipped[:, 6 + 32 * 2 : 6 + 32 * 3]

    ### Then, average the two outputs to get the "symmetric" result
    y_fused = (y + y_unflipped) / 2

    ### Unpack outputs
    analysis_confidence = _sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable
    CL = y_fused[:, 1] / 2
    CD = torch.exp((y_fused[:, 2] - 2) * 2)
    CM = y_fused[:, 3] / 20
    Top_Xtr = torch.clamp(y_fused[:, 4], 0, 1)  # Top_Xtr
    Bot_Xtr = torch.clamp(y_fused[:, 5], 0, 1)  # Bot_Xtr

    upper_bl_ue_over_vinf = y_fused[:, 6 + Data.N * 2 : 6 + Data.N * 3]
    lower_bl_ue_over_vinf = y_fused[:, 6 + Data.N * 5 : 6 + Data.N * 6]

    Re_col = Re.reshape(-1, 1)
    upper_theta = ((10 ** y_fused[:, 6 : 6 + Data.N]) - 0.1) / (
        torch.abs(upper_bl_ue_over_vinf) * Re_col
    )
    upper_H = 2.6 * torch.exp(y_fused[:, 6 + Data.N : 6 + Data.N * 2])

    lower_theta = ((10 ** y_fused[:, 6 + Data.N * 3 : 6 + Data.N * 4]) - 0.1) / (
        torch.abs(lower_bl_ue_over_vinf) * Re_col
    )
    lower_H = 2.6 * torch.exp(y_fused[:, 6 + Data.N * 4 : 6 + Data.N * 5])

    results = {
        "analysis_confidence": analysis_confidence,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        **{f"upper_bl_theta_{i}": upper_theta[:, i] for i in range(Data.N)},
        **{f"upper_bl_H_{i}": upper_H[:, i] for i in range(Data.N)},
        **{f"upper_bl_ue/vinf_{i}": upper_bl_ue_over_vinf[:, i] for i in range(Data.N)},
        **{f"lower_bl_theta_{i}": lower_theta[:, i] for i in range(Data.N)},
        **{f"lower_bl_H_{i}": lower_H[:, i] for i in range(Data.N)},
        **{f"lower_bl_ue/vinf_{i}": lower_bl_ue_over_vinf[:, i] for i in range(Data.N)},
    }
    return {key: value.reshape(-1) for key, value in results.items()}

def get_aero_from_airfoil_cuda(
    airfoil: Union[asb.Airfoil, asb.KulfanAirfoil],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size="xlarge",
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    
    ### Normalize the inputs and evaluate
    normalization_outputs = airfoil.normalize(return_dict=True)
    delta_alpha = normalization_outputs["rotation_angle"]  # degrees
    x_translation_LE = normalization_outputs["x_translation"]
    y_translation_LE = normalization_outputs["y_translation"]
    scale = normalization_outputs["scale_factor"]

    x_translation_qc = (
        -x_translation_LE + 0.25 * (1 / scale * np.cosd(delta_alpha)) - 0.25
    )
    y_translation_qc = -y_translation_LE + 0.25 * (1 / scale * np.sind(-delta_alpha))
    
    ### handle cuda variables
    normalized_airfoil = cuKulfanAirfoil(
        normalization_outputs["airfoil"].to_kulfan_airfoil(
            n_weights_per_side=8,
            normalize_coordinates=False,
        ),
        device=device
    )
    alpha = torch.tensor(alpha + delta_alpha, dtype=torch.float32, device=device)
    Re = torch.tensor(Re, dtype=torch.float32, device=device)
    n_crit = torch.tensor(n_crit, dtype=torch.float32, device=device)
    xtr_upper = torch.tensor(xtr_upper, dtype=torch.float32, device=device)
    xtr_lower = torch.tensor(xtr_lower, dtype=torch.float32, device=device)
        
    raw_aero = get_aero_from_kulfan_parameters_cuda(
        normalized_airfoil.kulfan_parameters_cuda,
        alpha,
        Re,
        n_crit,
        xtr_upper,
        xtr_lower,
        model_size,
        device
    ) 
    
    ### Correct the force vectors and lift-induced moment from translation
    extra_CM = -raw_aero["CL"] * x_translation_qc + raw_aero["CD"] * y_translation_qc
    raw_aero["CM"] = raw_aero["CM"] + extra_CM

    return raw_aero