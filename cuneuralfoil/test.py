import torch
import neuralfoil as nf
import aerosandbox as asb
import numpy as np
from cuneuralfoil.cu_kulfan_airfoil import cuKulfanAirfoil
from cuneuralfoil.main import get_aero_from_airfoil_cuda, get_aero_from_kulfan_parameters_cuda
import matplotlib.pyplot as plt
import itertools

device = "cuda"

# ----------------------- batched test on output consistency ----------------------- #
airfoil_names = ["naca4412", "PW75", "a63a108c", "naca3426"]
alpha_s = [-22.1, 0.0, 3.5, 60.6]
Re_s = [5e6, 1e3, 7e8]

def print_test_result(aero_1, aero_2):
    print(
        f"  CL: {aero_1['CL'].item():.4f}, {aero_2['CL'].item():.4f}, "
        f"diff={aero_1['CL'].item() - aero_2['CL'].item():.4f}"
    )
    print(
        f"  CD: {aero_1['CD'].item():.4f}, {aero_2['CD'].item():.4f}, "
        f"diff={aero_1['CD'].item() - aero_2['CD'].item():.4f}"
    )
    print(
        f"  CM: {aero_1['CM'].item():.4f}, {aero_2['CM'].item():.4f}, "
        f"diff={aero_1['CM'].item() - aero_2['CM'].item():.4f}"
    )
    print(
        f"  Top_Xtr: {aero_1['Top_Xtr'].item():.4f}, {aero_2['Top_Xtr'].item():.4f}, "
        f"diff={aero_1['Top_Xtr'].item() - aero_2['Top_Xtr'].item():.4f}"
    )
    print(
        f"  Bot_Xtr: {aero_1['Bot_Xtr'].item():.4f}, {aero_2['Bot_Xtr'].item():.4f}, "
        f"diff={aero_1['Bot_Xtr'].item() - aero_2['Bot_Xtr'].item():.4f}"
    )
    print(
        f"  confidence: {aero_1['analysis_confidence'].item():.4f}, "
        f"{aero_2['analysis_confidence'].item():.4f}, "
        f"diff={aero_1['analysis_confidence'].item() - aero_2['analysis_confidence'].item():.4f}"
    )

for airfoil_name, alpha, Re in itertools.product(airfoil_names, alpha_s, Re_s):
    airfoil = asb.Airfoil(airfoil_name).to_kulfan_airfoil()
    aero_1 = nf.get_aero_from_airfoil(airfoil=airfoil, alpha=alpha, Re=Re)
    aero_2 = get_aero_from_airfoil_cuda(airfoil=airfoil, alpha=alpha, Re=Re, device=device)
    print(f'testing {airfoil_name}, AoA={alpha}, Re={Re} ---------------------------------------')
    print_test_result(aero_1, aero_2)

# ----------------------- case comparison on differentiability ----------------------- #
Re = 5e6
alpha = 3

def finite_difference(f, kulfan_parameters, alpha, Re, eps: float = 1e-6, model_size: str = 'xlarge'):
    unpacked_var = np.concatenate([
        kulfan_parameters["lower_weights"], 
        kulfan_parameters["upper_weights"],
        np.atleast_1d(alpha).astype(float),
        np.atleast_1d(Re).astype(float)
    ])
    FD_dCL_Dx = np.zeros_like(unpacked_var)
    FD_dCD_Dx = np.zeros_like(unpacked_var)
    FD_dCM_Dx = np.zeros_like(unpacked_var)
    for i in range(unpacked_var.size):
        var_plus = unpacked_var.copy()
        var_plus[i] += eps
        var_minus = unpacked_var.copy()
        var_minus[i] -= eps
        f_plus = f({
            "lower_weights": var_plus[:8],
            "upper_weights": var_plus[8:16],
            "leading_edge_weight": kulfan_parameters["leading_edge_weight"],
            "TE_thickness": kulfan_parameters["TE_thickness"],
        }, var_plus[-2], var_plus[-1])
        f_minus = f({
            "lower_weights": var_minus[:8],
            "upper_weights": var_minus[8:16],
            "leading_edge_weight": kulfan_parameters["leading_edge_weight"],
            "TE_thickness": kulfan_parameters["TE_thickness"],
        }, var_minus[-2], var_minus[-1])
        FD_dCL_Dx[i] = (f_plus['CL'].squeeze() - f_minus['CL'].squeeze()) / (2 * eps)
        FD_dCD_Dx[i] = (f_plus['CD'].squeeze() - f_minus['CD'].squeeze()) / (2 * eps)
        FD_dCM_Dx[i] = (f_plus['CM'].squeeze() - f_minus['CM'].squeeze()) / (2 * eps)
        
    FD_dCL_dlower = FD_dCL_Dx[:8]
    FD_dCL_dupper = FD_dCL_Dx[8:16]
    FD_dCL_dalpha = FD_dCL_Dx[-2]
    FD_dCL_dRe = FD_dCL_Dx[-1]
    
    FD_dCD_dlower = FD_dCD_Dx[:8]
    FD_dCD_dupper = FD_dCD_Dx[8:16]
    FD_dCD_dalpha = FD_dCD_Dx[-2]
    FD_dCD_dRe = FD_dCD_Dx[-1]
    
    FD_dCM_dlower = FD_dCM_Dx[:8]
    FD_dCM_dupper = FD_dCM_Dx[8:16]
    FD_dCM_dalpha = FD_dCM_Dx[-2]
    FD_dCM_dRe = FD_dCM_Dx[-1]
    
    return [
        [FD_dCL_dlower, FD_dCL_dupper, FD_dCL_dalpha, FD_dCL_dRe],
        [FD_dCD_dlower, FD_dCD_dupper, FD_dCD_dalpha, FD_dCD_dRe],
        [FD_dCM_dlower, FD_dCM_dupper, FD_dCM_dalpha, FD_dCM_dRe],
    ]

def plot_fd_vs_ad_gradients(
    FD_dCL_dlower, AD_dCL_dlower,
    FD_dCL_dupper, AD_dCL_dupper,
    FD_dCD_dlower, AD_dCD_dlower,
    FD_dCD_dupper, AD_dCD_dupper,
    FD_dCM_dlower, AD_dCM_dlower,
    FD_dCM_dupper, AD_dCM_dupper,
):
    # Convert everything to numpy
    AD_dCL_dlower = AD_dCL_dlower.cpu().data.numpy()
    AD_dCL_dupper = AD_dCL_dupper.cpu().data.numpy()
    AD_dCD_dlower = AD_dCD_dlower.cpu().data.numpy()
    AD_dCD_dupper = AD_dCD_dupper.cpu().data.numpy()
    AD_dCM_dlower = AD_dCM_dlower.cpu().data.numpy()
    AD_dCM_dupper = AD_dCM_dupper.cpu().data.numpy()

    # Sanity check length
    n = FD_dCL_dlower.shape[0]
    x = np.arange(n)
    bar_width = 0.4

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=False)
    axes = axes.ravel()

    data_pairs = [
        (FD_dCL_dlower, AD_dCL_dlower, r"gradients of CL w.r.t. lower weights"),
        (FD_dCL_dupper, AD_dCL_dupper, r"gradients of CL w.r.t. upper weights"),
        (FD_dCD_dlower, AD_dCD_dlower, r"gradients of CD w.r.t. lower weights"),
        (FD_dCD_dupper, AD_dCD_dupper, r"gradients of CD w.r.t. upper weights"),
        (FD_dCM_dlower, AD_dCM_dlower, r"gradients of CM w.r.t. lower weights"),
        (FD_dCM_dupper, AD_dCM_dupper, r"gradients of CM w.r.t. upper weights"),
    ]

    for ax, (fd, ad, title) in zip(axes, data_pairs):
        ax.bar(x - bar_width / 2, fd, width=bar_width, label="Finite Difference (FD)", alpha=0.8)
        ax.bar(x + bar_width / 2, ad, width=bar_width, label="Auto Differentiation (AD)", alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("Gradient value")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}" for i in x])

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Put a single legend at the top center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for global legend
    plt.savefig('test_gradients.png')
    plt.close()
        

airfoil = asb.Airfoil("naca4412").to_kulfan_airfoil()
FD_results = finite_difference(nf.get_aero_from_kulfan_parameters, airfoil.kulfan_parameters, alpha, Re)
FD_dCL_dlower, FD_dCL_dupper, FD_dCL_dalpha, FD_dCL_dRe = FD_results[0]
FD_dCD_dlower, FD_dCD_dupper, FD_dCD_dalpha, FD_dCD_dRe = FD_results[1]
FD_dCM_dlower, FD_dCM_dupper, FD_dCM_dalpha, FD_dCM_dRe = FD_results[2]

airfoil = cuKulfanAirfoil(
    airfoil,
    requires_grad = True,
    device = device
)
alpha = torch.tensor(alpha, dtype=torch.float32, device=device, requires_grad=True)
Re = torch.tensor(Re, dtype=torch.float32, device=device, requires_grad=True)
aero = get_aero_from_kulfan_parameters_cuda(airfoil.kulfan_parameters_cuda, alpha, Re)
AD_dCL_dlower = torch.autograd.grad(aero['CL'], airfoil.lower_weights_cuda, retain_graph=True)[0]
AD_dCL_dupper = torch.autograd.grad(aero['CL'], airfoil.upper_weights_cuda, retain_graph=True)[0]
AD_dCL_dalpha = torch.autograd.grad(aero['CL'], alpha, retain_graph=True)[0]
AD_dCL_dRe = torch.autograd.grad(aero['CL'], Re, retain_graph=True)[0]
AD_dCD_dlower = torch.autograd.grad(aero['CD'], airfoil.lower_weights_cuda, retain_graph=True)[0]
AD_dCD_dupper = torch.autograd.grad(aero['CD'], airfoil.upper_weights_cuda, retain_graph=True)[0]
AD_dCD_dalpha = torch.autograd.grad(aero['CD'], alpha, retain_graph=True)[0]
AD_dCD_dRe = torch.autograd.grad(aero['CD'], Re, retain_graph=True)[0]
AD_dCM_dlower = torch.autograd.grad(aero['CM'], airfoil.lower_weights_cuda, retain_graph=True)[0]
AD_dCM_dupper = torch.autograd.grad(aero['CM'], airfoil.upper_weights_cuda, retain_graph=True)[0]
AD_dCM_dalpha = torch.autograd.grad(aero['CM'], alpha, retain_graph=True)[0]
AD_dCM_dRe = torch.autograd.grad(aero['CM'], Re)[0]

print(f'testing {airfoil_name}\'s gradients, AoA={alpha}, Re={Re} ---------------------------------------')
print(f'  FD_dCL_dAOA={FD_dCL_dalpha:.4e},\t AD_dCL_dAOA={AD_dCL_dalpha:.4e}')
print(f'  FD_dCL_dRe={FD_dCL_dRe:.4e},\t AD_dCL_dRe={AD_dCL_dRe:.4e}')
print(f'  FD_dCD_dAOA={FD_dCD_dalpha:.4e},\t AD_dCD_dAOA={AD_dCD_dalpha:.4e}')
print(f'  FD_dCD_dRe={FD_dCD_dRe:.4e},\t AD_dCD_dRe={AD_dCD_dRe:.4e}')
print(f'  FD_dCM_dAOA={FD_dCM_dalpha:.4f},\t AD_dCM_dAOA={AD_dCM_dalpha:.4e}')
print(f'  FD_dCM_dRe={FD_dCM_dRe:.4e},\t AD_dCM_dRe={AD_dCM_dRe:.4e}')

plot_fd_vs_ad_gradients(
    FD_dCL_dlower, AD_dCL_dlower,
    FD_dCL_dupper, AD_dCL_dupper,
    FD_dCD_dlower, AD_dCD_dlower,
    FD_dCD_dupper, AD_dCD_dupper,
    FD_dCM_dlower, AD_dCM_dlower,
    FD_dCM_dupper, AD_dCM_dupper,
)
print('  Generate gradient comparison of CST parameters in test_gradients.png')


# ----------------------- runtime comparison on batched prediction ----------------------- #
import time
B = 1000  # batch size
Re = 5e6
alpha = 3
model_sizes = ["xxsmall", "small", "medium", "large", "xlarge", "xxxlarge"]
result = {}

def plot_timing(results: dict):
    n_models = len(model_sizes)
    x = np.arange(n_models)

    # Extract data
    time_nf_forward = np.array([results[m][0] for m in model_sizes])
    time_nf_diff    = np.array([results[m][1] for m in model_sizes])
    time_cu_forward = np.array([results[m][2] for m in model_sizes])
    time_cu_diff    = np.array([results[m][3] for m in model_sizes])
    forward_speedup = np.array([results[m][4] for m in model_sizes])
    diff_speedup    = np.array([results[m][5] for m in model_sizes])

    fig, (ax_fwd, ax_diff) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    bar_width = 0.3
    
    # Top subplot forward runtime
    b1 = ax_fwd.bar(x - bar_width / 2, time_nf_forward, width=bar_width, label="NF forward")
    b2 = ax_fwd.bar(x + bar_width / 2, time_cu_forward, width=bar_width, label="cuNF forward")
    ax_fwd.set_ylabel("Forward time [s]")
    ax_fwd.set_title("NeuralFoil vs cuNeuralFoil runtime and speedup")
    ax_fwd.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # ax_fwd.set_yscale("log")
    # Line plot for speedup
    ax_fwd_speed = ax_fwd.twinx()
    l1, = ax_fwd_speed.plot(
        x,
        forward_speedup,
        marker="o",
        linewidth=2,
        color="tab:red",
        label="Forward speedup",
    )
    ax_fwd_speed.set_ylabel("Forward runtime speedup")
    ax_fwd_speed.grid(False)
    # Legend for forward subplot
    handles_fwd = [b1, b2, l1]
    labels_fwd = [h.get_label() for h in handles_fwd]
    ax_fwd.legend(handles_fwd, labels_fwd, loc="upper left")
    ax_fwd.set_xticklabels([])

    # Bottom subplot diff runtime
    b3 = ax_diff.bar(x - bar_width / 2, time_nf_diff, width=bar_width, label="NF diff")
    b4 = ax_diff.bar(x + bar_width / 2, time_cu_diff, width=bar_width, label="cuNF diff")
    ax_diff.set_ylabel("Differetiation time [s]")
    ax_diff.set_xlabel("Model size")
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(model_sizes)
    ax_diff.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # ax_diff.set_yscale("log")
    # Line plot for speedup
    ax_diff_speed = ax_diff.twinx()
    l2, = ax_diff_speed.plot(
        x,
        diff_speedup,
        marker="s",
        linewidth=2,
        color="tab:green",
        label="Differentiation speedup",
    )
    ax_diff_speed.set_ylabel("Differentiation runtime speedup")
    ax_diff_speed.grid(False)
    handles_diff = [b3, b4, l2]
    labels_diff = [h.get_label() for h in handles_diff]
    ax_diff.legend(handles_diff, labels_diff, loc="upper left")

    plt.setp(ax_diff.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig('test_runtime.png')
    plt.close()

for model_size in model_sizes:
    # batched neural foil forward
    time1 = time.perf_counter()
    airfoil = asb.Airfoil(airfoil_name).to_kulfan_airfoil()
    for _ in range(B):
        aero = nf.get_aero_from_kulfan_parameters(airfoil.kulfan_parameters, alpha, Re, model_size=model_size)

    # batched neural foil FD on lower_weight, upper_weight, alpha and Re
    time2 = time.perf_counter()
    for _ in range(B):
        FD_results = finite_difference(nf.get_aero_from_kulfan_parameters, airfoil.kulfan_parameters, alpha, Re, model_size=model_size)
    time3 = time.perf_counter()

    # cuda neural foil
    airfoil = cuKulfanAirfoil(
        airfoil,
        requires_grad = True,
        device = device
    )
    upper_batch = airfoil.upper_weights_cuda.unsqueeze(0).repeat(B, 1)
    lower_batch = airfoil.lower_weights_cuda.unsqueeze(0).repeat(B, 1)
    LE_batch = airfoil.leading_edge_weight_cuda.repeat(B)
    TE_batch = airfoil.TE_thickness_cuda.repeat(B)
    kulfan_batch = {
        "upper_weights_cuda": upper_batch.to(device),
        "lower_weights_cuda": lower_batch.to(device),
        "leading_edge_weight_cuda": LE_batch.to(device),
        "TE_thickness_cuda": TE_batch.to(device),
    }
    alpha_batch = torch.full((B,), alpha, dtype=torch.float32, device=device, requires_grad=True)
    Re_batch = torch.full((B,), Re, dtype=torch.float32, device=device, requires_grad=True)
    ones = torch.ones((B,), dtype=torch.float32, device=device)

    # batched cuda neural foil forward
    time4 = time.perf_counter()
    aero = get_aero_from_kulfan_parameters_cuda(kulfan_batch, alpha_batch, Re_batch, device=device, model_size=model_size)
    torch.cuda.synchronize()

    # batched neural foil FD on lower_weight, upper_weight, alpha and Re
    time5 = time.perf_counter()
    aero = get_aero_from_kulfan_parameters_cuda(kulfan_batch, alpha_batch, Re_batch, device=device, model_size=model_size)
    AD_dCL_dlower = torch.autograd.grad(aero['CL'], lower_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCL_dupper = torch.autograd.grad(aero['CL'], upper_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCL_dalpha = torch.autograd.grad(aero['CL'], alpha_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCL_dRe = torch.autograd.grad(aero['CL'], Re_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCD_dlower = torch.autograd.grad(aero['CD'], lower_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCD_dupper = torch.autograd.grad(aero['CD'], upper_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCD_dalpha = torch.autograd.grad(aero['CD'], alpha_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCD_dRe = torch.autograd.grad(aero['CD'], Re_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCM_dlower = torch.autograd.grad(aero['CM'], lower_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCM_dupper = torch.autograd.grad(aero['CM'], upper_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCM_dalpha = torch.autograd.grad(aero['CM'], alpha_batch, grad_outputs=ones, retain_graph=True)[0]
    AD_dCM_dRe = torch.autograd.grad(aero['CM'], Re_batch, grad_outputs=ones, retain_graph=True)[0]
    torch.cuda.synchronize()
    time6 = time.perf_counter()
    
    time_nf_forward = time2 - time1
    time_nf_diff = time3 - time2
    time_cunf_forward = time5 - time4
    time_cunf_diff = time6 - time5
    forward_speedup = time_nf_forward/time_cunf_forward
    diff_speedup = time_nf_diff/time_cunf_diff
    
    result[f"{model_size}"] = [time_nf_forward, time_nf_diff, time_cunf_forward, time_cunf_diff, forward_speedup, diff_speedup]

    print(f'testing computational time costs of model {model_size}, batch size={B} -------------------------')
    print(f'  {B} neural foil prediction: {time_nf_forward:.4f}s')
    print(f'  {B} neural foil differentiation (w.r,t. lower/upper weights, AoA, Re): {time_nf_diff:.4f}s')
    print(f'  {B} cuda neural foil prediction: {time_cunf_forward:.4f}s')
    print(f'  {B} cuda neural foil differentiation (w.r,t. lower/upper weights, AoA, Re): {time_cunf_diff:.4f}s')
    print(f'  inference speedup: {forward_speedup:.2f}, differentiation speedup: {diff_speedup:.2f}')

plot_timing(result)
print(f'  Computation time comparison saves in test_runtime.png')