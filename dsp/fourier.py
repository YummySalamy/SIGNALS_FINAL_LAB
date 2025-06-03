"""
Fourier Series Analysis Module

Implements the computation of Fourier series coefficients and signal reconstruction
for the classic textbook examples 3.6.1 through 3.6.4.

Mathematical Foundation:
    Trigonometric Fourier Series:
    x(t) = a₀/2 + Σ[aₖcos(kω₀t) + bₖsin(kω₀t)]
    
    where:
    a₀ = (2/T)∫₀ᵀ x(t)dt
    aₖ = (2/T)∫₀ᵀ x(t)cos(kω₀t)dt  
    bₖ = (2/T)∫₀ᵀ x(t)sin(kω₀t)dt
    
    Exponential form:
    Cₖ = (1/T)∫₀ᵀ x(t)e^(-jkω₀t)dt
    x(t) = Σ Cₖe^(jkω₀t)
"""

import numpy as np
from scipy import integrate
from typing import Dict, Tuple, List
import warnings

def get_signal_parameters(example_id: str) -> Dict:
    """
    Get the mathematical parameters for each textbook example.
    
    Args:
        example_id: Signal identifier ("3.6.1", "3.6.2", "3.6.3", "3.6.4")
        
    Returns:
        Dictionary with period, symmetry, and description
    """
    params = {
        "3.6.1": {
            "period": 2.0,
            "symmetry": "even",  # Only cosine terms (bₖ = 0)
            "description": "Triangular wave: x(t) = 1 - |t|",
            "analytical_coeffs": lambda k: 8/(np.pi**2 * k**2) if k % 2 == 1 else 0
        },
        "3.6.2": {
            "period": 2*np.pi,
            "symmetry": "odd",   # Only sine terms (aₖ = a₀ = 0)
            "description": "Sawtooth wave: x(t) = t",
            "analytical_coeffs": lambda k: 2*(-1)**(k+1)/k
        },
        "3.6.3": {
            "period": 2*np.pi,
            "symmetry": "even",  # Only cosine terms
            "description": "Parabolic wave: x(t) = t²",
            "analytical_coeffs": lambda k: 4*(-1)**k/(k**2)
        },
        "3.6.4": {
            "period": 2.0,
            "symmetry": "none",  # General case
            "description": "Mixed ramp+step: x(t) = t+1 for t<0, x(t) = 1 for t>0",
            "analytical_coeffs": None  # No simple closed form
        }
    }
    
    if example_id not in params:
        raise ValueError(f"Unknown example: {example_id}")
        
    return params[example_id]

def eval_3_6_signal(example_id: str, t: np.ndarray) -> np.ndarray:
    """
    Evaluate the analytical signal for textbook examples 3.6.1-3.6.4.
    
    Args:
        example_id: Signal identifier
        t: Time vector (should be in range [-T/2, T/2])
        
    Returns:
        Signal values x(t)
    """
    if example_id == "3.6.1":
        # Triangular: x(t) = 1 - |t| for |t| ≤ 1
        return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)
        
    elif example_id == "3.6.2":
        # Sawtooth: x(t) = t for |t| ≤ π
        return np.where(np.abs(t) <= np.pi, t, 0)
        
    elif example_id == "3.6.3":
        # Parabolic: x(t) = t² for |t| ≤ π
        return np.where(np.abs(t) <= np.pi, t**2, 0)
        
    elif example_id == "3.6.4":
        # Mixed: piecewise function
        return np.where(t < 0, t + 1, np.where(t < 1, 1, 0))
        
    else:
        raise ValueError(f"Unknown example: {example_id}")

def eval_3_6_signal_periodic(example_id: str, t: np.ndarray, T: float) -> np.ndarray:
    """
    Evaluate the periodic extension of the signal.
    
    Args:
        example_id: Signal identifier
        t: Time vector (can span multiple periods)
        T: Fundamental period
        
    Returns:
        Periodic signal values
    """
    # Map to fundamental period [-T/2, T/2]
    t_mod = ((t + T/2) % T) - T/2
    return eval_3_6_signal(example_id, t_mod)

def compute_fourier_coefficients(example_id: str, N: int) -> Dict:
    """
    Compute Fourier series coefficients using numerical integration.
    
    Args:
        example_id: Signal identifier
        N: Number of harmonics to compute
        
    Returns:
        Dictionary with coefficients and metadata
        
    Mathematical Note:
        Uses Simpson's rule for numerical integration with high precision.
        For even/odd signals, exploits symmetry to reduce computation.
    """
    params = get_signal_parameters(example_id)
    T = params["period"]
    omega0 = 2*np.pi/T
    
    # High-resolution time vector for integration
    t_int = np.linspace(-T/2, T/2, 8192, endpoint=False)
    x_int = eval_3_6_signal(example_id, t_int)
    
    # Initialize coefficient arrays
    a_coeffs = np.zeros(N+1)  # a₀, a₁, ..., aₙ
    b_coeffs = np.zeros(N+1)  # b₀, b₁, ..., bₙ (b₀ always 0)
    c_coeffs = np.zeros(2*N+1, dtype=complex)  # C₋ₙ, ..., C₀, ..., Cₙ
    
    # DC component a₀
    if params["symmetry"] != "odd":
        a_coeffs[0] = 2/T * np.trapz(x_int, t_int)
    
    # Harmonic components
    for k in range(1, N+1):
        if params["symmetry"] != "odd":
            # Cosine coefficients aₖ
            cos_k = np.cos(k * omega0 * t_int)
            a_coeffs[k] = 2/T * np.trapz(x_int * cos_k, t_int)
            
        if params["symmetry"] != "even":
            # Sine coefficients bₖ  
            sin_k = np.sin(k * omega0 * t_int)
            b_coeffs[k] = 2/T * np.trapz(x_int * sin_k, t_int)
        
        # Complex exponential coefficients Cₖ
        exp_pos = np.exp(-1j * k * omega0 * t_int)
        exp_neg = np.exp(1j * k * omega0 * t_int)
        
        c_coeffs[N + k] = 1/T * np.trapz(x_int * exp_pos, t_int)  # Cₖ
        c_coeffs[N - k] = 1/T * np.trapz(x_int * exp_neg, t_int)  # C₋ₖ
    
    # DC component for complex form
    c_coeffs[N] = a_coeffs[0] / 2
    
    return {
        "a_coeffs": a_coeffs,
        "b_coeffs": b_coeffs,
        "c_coeffs": c_coeffs,
        "N": N,
        "T": T,
        "omega0": omega0,
        "symmetry": params["symmetry"],
        "description": params["description"]
    }

def reconstruct_fourier_series(coeffs: Dict, t: np.ndarray, T: float = None) -> np.ndarray:
    """
    Reconstruct signal from Fourier series coefficients.
    
    Args:
        coeffs: Coefficient dictionary from compute_fourier_coefficients
        t: Time vector for reconstruction
        T: Period (if None, use from coeffs)
        
    Returns:
        Reconstructed signal x̂(t)
        
    Mathematical Note:
        x̂(t) = a₀/2 + Σₖ₌₁ᴺ [aₖcos(kω₀t) + bₖsin(kω₀t)]
    """
    if T is None:
        T = coeffs["T"]
    
    omega0 = 2*np.pi/T
    N = coeffs["N"]
    
    # Start with DC component
    x_recon = np.full_like(t, coeffs["a_coeffs"][0] / 2, dtype=float)
    
    # Add harmonic components
    for k in range(1, N+1):
        # Cosine term
        if abs(coeffs["a_coeffs"][k]) > 1e-12:
            x_recon += coeffs["a_coeffs"][k] * np.cos(k * omega0 * t)
            
        # Sine term  
        if abs(coeffs["b_coeffs"][k]) > 1e-12:
            x_recon += coeffs["b_coeffs"][k] * np.sin(k * omega0 * t)
    
    return x_recon

def compute_rms_error(x_original: np.ndarray, x_reconstructed: np.ndarray) -> float:
    """
    Compute RMS error between original and reconstructed signals.
    
    Args:
        x_original: Original signal
        x_reconstructed: Reconstructed signal
        
    Returns:
        RMS error value
        
    Mathematical Note:
        RMS = √[(1/N)Σ(x - x̂)²]
    """
    if len(x_original) != len(x_reconstructed):
        min_len = min(len(x_original), len(x_reconstructed))
        x_original = x_original[:min_len]
        x_reconstructed = x_reconstructed[:min_len]
    
    error = x_original - x_reconstructed
    return np.sqrt(np.mean(error**2))

def compute_gibbs_overshoot(example_id: str, N: int) -> float:
    """
    Compute Gibbs phenomenon overshoot for discontinuous signals.
    
    Args:
        example_id: Signal identifier
        N: Number of harmonics
        
    Returns:
        Percentage overshoot due to Gibbs phenomenon
        
    Note:
        For signals with jump discontinuities, the Fourier series exhibits
        ~9% overshoot that doesn't decrease with increasing N.
    """
    if example_id not in ["3.6.2", "3.6.4"]:
        return 0.0  # No discontinuities
    
    params = get_signal_parameters(example_id)
    T = params["period"]
    
    # Evaluate near discontinuity
    coeffs = compute_fourier_coefficients(example_id, N)
    
    if example_id == "3.6.2":
        # Sawtooth: discontinuity at t = ±π
        t_test = np.linspace(np.pi - 0.1, np.pi + 0.1, 1000)
        x_true = eval_3_6_signal_periodic(example_id, t_test, T)
        x_recon = reconstruct_fourier_series(coeffs, t_test, T)
        
        # Find maximum overshoot
        jump_magnitude = 2*np.pi  # Jump size
        overshoot = np.max(np.abs(x_recon - x_true)) / jump_magnitude * 100
        
    elif example_id == "3.6.4":
        # Mixed signal: discontinuity at t = 0
        t_test = np.linspace(-0.1, 0.1, 1000)
        x_true = eval_3_6_signal_periodic(example_id, t_test, T)
        x_recon = reconstruct_fourier_series(coeffs, t_test, T)
        
        overshoot = np.max(np.abs(x_recon - x_true)) * 100
        
    return overshoot

def get_convergence_analysis(example_id: str, N_max: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze convergence of Fourier series reconstruction.
    
    Args:
        example_id: Signal identifier
        N_max: Maximum number of harmonics to test
        
    Returns:
        Tuple of (N_values, rms_errors)
    """
    params = get_signal_parameters(example_id)
    T = params["period"]
    
    # Reference signal (high resolution)
    t_ref = np.linspace(-T/2, T/2, 4096, endpoint=False)
    x_ref = eval_3_6_signal(example_id, t_ref)
    
    N_values = np.arange(1, N_max + 1)
    rms_errors = []
    
    for N in N_values:
        coeffs = compute_fourier_coefficients(example_id, N)
        x_recon = reconstruct_fourier_series(coeffs, t_ref, T)
        rms_error = compute_rms_error(x_ref, x_recon)
        rms_errors.append(rms_error)
    
    return N_values, np.array(rms_errors)