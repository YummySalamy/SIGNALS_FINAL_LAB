"""
Digital Modulation and Demodulation Module

Implements DSB-SC, I/Q (QAM), and DSB-LC amplitude modulation schemes
as used in communication systems.

Mathematical Foundations:

1. DSB-SC (Double Sideband Suppressed Carrier):
   TX: y(t) = x(t) × cos(ωct)
   Spectrum: Y(f) = ½[X(f-fc) + X(f+fc)]
   RX: x̂(t) = LPF{2y(t) × cos(ωct)}

2. I/Q Modulation (Quadrature Amplitude Modulation):
   TX: s(t) = xI(t)cos(ωct) + xQ(t)sin(ωct)  
   RX: xI(t) = LPF{2s(t) × cos(ωct)}
       xQ(t) = LPF{2s(t) × sin(ωct)}
   Orthogonality: ∫cos(ωct)sin(ωct)dt = 0

3. DSB-LC (Double Sideband Large Carrier / AM):
   TX: s(t) = [1 + μm(t)] × cos(ωct)
   μ > 1 causes overmodulation and distortion
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings

def dsb_sc_modulate(x: np.ndarray, fc: float, fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    DSB-SC modulation with perfect temporal synchronization.
    
    Args:
        x: Message signal (baseband)
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        
    Returns:
        Tuple of (modulated_signal, synchronized_message, final_fs)
        
    Mathematical Note:
        Ensures perfect temporal alignment between message and carrier
        by using unified time grid for both signals.
    """
    if fc >= fs/2:
        warnings.warn(f"Carrier frequency {fc} Hz exceeds Nyquist limit {fs/2} Hz")
    
    # Check if we need higher sampling frequency
    fs_min_required = 2.5 * fc  # Factor de seguridad
    
    if fs < fs_min_required:
        # Necesitamos una frecuencia más alta
        fs_new = fs_min_required
        
        # Crear vector temporal original
        t_orig = np.arange(len(x)) / fs
        duration = t_orig[-1]
        
        # Nuevo vector temporal de alta resolución
        t_new = np.arange(0, duration, 1/fs_new)
        
        # Interpolar señal mensaje al nuevo grid temporal
        x_interp = np.interp(t_new, t_orig, x)
        
        # Generar portadora en el MISMO grid temporal
        carrier = np.cos(2 * np.pi * fc * t_new)
        
        # Modulación con vectores perfectamente sincronizados
        y_modulated = x_interp * carrier
        
        return y_modulated, x_interp, fs_new
    else:
        # Si fs ya es suficiente, usar sincronización normal
        t = np.arange(len(x)) / fs
        carrier = np.cos(2 * np.pi * fc * t)
        y_modulated = x * carrier
        
        return y_modulated, x, fs

def dsb_sc_demod_mix(y: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    DSB-SC demodulation: coherent mixing stage.
    
    Args:
        y: Received modulated signal
        fc: Carrier frequency (Hz) - must match TX exactly
        fs: Sampling frequency (Hz)
        
    Returns:
        Mixed signal before filtering: 2y(t) × cos(ωct)
        
    Mathematical Note:
        Product: 2x(t)cos²(ωct) = x(t)[1 + cos(2ωct)]
        Contains baseband x(t) + replica at 2fc
    """
    t = np.arange(len(y)) / fs
    local_osc = 2 * np.cos(2 * np.pi * fc * t)  # Factor of 2 for gain compensation
    
    return y * local_osc

def dsb_sc_demodulate(y: np.ndarray, fc: float, fs: float, sos: np.ndarray) -> np.ndarray:
    """
    Complete DSB-SC demodulation with filtering.
    
    Args:
        y: Received modulated signal
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        sos: Second-order sections filter coefficients
        
    Returns:
        Demodulated baseband signal x̂(t)
    """
    # Coherent mixing
    mixed = dsb_sc_demod_mix(y, fc, fs)
    
    # Low-pass filtering to remove 2fc component
    demodulated = signal.sosfilt(sos, mixed)
    
    return demodulated

def iq_modulate(xI: np.ndarray, xQ: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    I/Q (Quadrature) modulation for simultaneous transmission.
    
    Args:
        xI: In-phase message signal
        xQ: Quadrature message signal  
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        
    Returns:
        Modulated signal s(t) = xI(t)cos(ωct) + xQ(t)sin(ωct)
        
    Mathematical Note:
        Exploits orthogonality of sine and cosine to transmit two
        independent signals simultaneously at the same frequency.
    """
    if len(xI) != len(xQ):
        min_len = min(len(xI), len(xQ))
        xI = xI[:min_len]
        xQ = xQ[:min_len]
        warnings.warn(f"I/Q signals length mismatch, truncated to {min_len}")
    
    t = np.arange(len(xI)) / fs
    
    # In-phase component
    I_modulated = xI * np.cos(2 * np.pi * fc * t)
    
    # Quadrature component  
    Q_modulated = xQ * np.sin(2 * np.pi * fc * t)
    
    return I_modulated + Q_modulated

def iq_demodulate(s: np.ndarray, fc: float, fs: float, sos: np.ndarray, 
                  phase_error: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    I/Q demodulation with optional phase error simulation.
    
    Args:
        s: Received I/Q modulated signal
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        sos: Low-pass filter coefficients
        phase_error: Phase error in radians (default: 0)
        
    Returns:
        Tuple of (xI_recovered, xQ_recovered)
        
    Mathematical Note:
        Phase error causes crosstalk between I and Q channels.
        Perfect recovery requires φ_error = 0.
    """
    t = np.arange(len(s)) / fs
    
    # Local oscillators with phase error
    cos_lo = 2 * np.cos(2 * np.pi * fc * t + phase_error)
    sin_lo = 2 * np.sin(2 * np.pi * fc * t + phase_error)
    
    # Mixing (coherent detection)
    I_mixed = s * cos_lo
    Q_mixed = s * sin_lo
    
    # Low-pass filtering to recover baseband
    xI_recovered = signal.sosfilt(sos, I_mixed)
    xQ_recovered = signal.sosfilt(sos, Q_mixed)
    
    return xI_recovered, xQ_recovered

def dsb_lc_modulate(m: np.ndarray, mu: float, fc: float, fs: float) -> np.ndarray:
    """
    DSB-LC (AM) modulation with large carrier.
    
    Args:
        m: Normalized message signal (range ±1)
        mu: Modulation index (μ)
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        
    Returns:
        AM signal s(t) = [1 + μm(t)] × cos(ωct)
        
    Mathematical Note:
        μ < 1: Normal modulation
        μ = 1: 100% modulation (envelope touches zero)
        μ > 1: Overmodulation (envelope inversion, distortion)
    """
    # Normalize message signal to ±1 range
    if np.max(np.abs(m)) > 0:
        m_norm = m / np.max(np.abs(m))
    else:
        m_norm = m
    
    # Check for overmodulation
    envelope = 1 + mu * m_norm
    if np.min(envelope) < 0:
        warnings.warn(f"Overmodulation detected: μ={mu:.2f}, min envelope={np.min(envelope):.3f}")
    
    t = np.arange(len(m)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    
    return envelope * carrier

def envelope_detector(s_am: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    Envelope detection for AM signals using rectification and low-pass filtering.
    
    Args:
        s_am: AM modulated signal
        fs: Sampling frequency (Hz)
        fc: Carrier frequency (Hz)
        
    Returns:
        Detected envelope (approximates 1 + μm(t))
        
    Mathematical Note:
        Simple envelope detector: |s(t)| → LPF
        More sophisticated: Hilbert transform method
    """
    # Rectification (absolute value)
    rectified = np.abs(s_am)
    
    # Low-pass filter to smooth envelope
    # Cutoff should be >> message BW but << carrier frequency
    cutoff = fc / 10  # Conservative choice
    sos = signal.butter(6, cutoff, btype='low', fs=fs, output='sos')
    
    envelope = signal.sosfilt(sos, rectified)
    
    return envelope

def hilbert_envelope_detector(s_am: np.ndarray) -> np.ndarray:
    """
    Envelope detection using Hilbert transform (ideal method).
    
    Args:
        s_am: AM modulated signal
        
    Returns:
        Instantaneous envelope |s(t) + jH{s(t)}|
        
    Mathematical Note:
        More accurate than rectifier-filter method.
        Envelope = |analytical_signal| = sqrt(s² + H{s}²)
    """
    analytical_signal = signal.hilbert(s_am)
    envelope = np.abs(analytical_signal)
    
    return envelope

def compute_modulation_index(envelope: np.ndarray) -> float:
    """
    Estimate modulation index from detected envelope.
    
    Args:
        envelope: Detected envelope signal
        
    Returns:
        Estimated modulation index μ
        
    Mathematical Note:
        μ = (Emax - Emin) / (Emax + Emin)
        where Emax, Emin are envelope extrema
    """
    # Remove DC component and find extrema
    envelope_ac = envelope - np.mean(envelope)
    
    if len(envelope_ac) == 0:
        return 0.0
    
    E_max = np.max(envelope_ac)
    E_min = np.min(envelope_ac)
    
    if (E_max + E_min) == 0:
        return 0.0
    
    mu_estimated = (E_max - E_min) / (E_max + E_min)
    
    return mu_estimated

def generate_demo_signal(signal_type: str, duration: float, fs: float) -> Tuple[np.ndarray, float]:
    """
    Generate demonstration signals for testing modulation schemes.
    
    Args:
        signal_type: Type of demo signal
        duration: Signal duration in seconds
        fs: Sampling frequency (Hz)
        
    Returns:
        Tuple of (signal, actual_fs)
    """
    t = np.arange(0, duration, 1/fs)
    
    if signal_type == "Tono puro" or signal_type == "Tono 1kHz":
        # Single tone at 1 kHz
        x = 0.8 * np.sin(2 * np.pi * 1000 * t)
        
    elif signal_type == "Tono 2kHz":
        # Single tone at 2 kHz (for Q channel)
        x = 0.8 * np.sin(2 * np.pi * 2000 * t)
        
    elif signal_type == "Suma de tonos" or signal_type == "Suma tonos":
        # Multi-tone signal
        x = (0.4 * np.sin(2 * np.pi * 800 * t) + 
             0.3 * np.sin(2 * np.pi * 1200 * t) + 
             0.2 * np.sin(2 * np.pi * 2000 * t))
        
    elif signal_type == "Chirp":
        # Linear frequency sweep
        f0, f1 = 100, 3000  # 100 Hz to 3 kHz
        x = 0.7 * signal.chirp(t, f0, duration, f1, method='linear')
        
    elif signal_type == "Ruido filtrado":
        # Band-limited white noise
        noise = np.random.randn(len(t))
        # Low-pass filter at 3 kHz
        sos = signal.butter(6, 3000, btype='low', fs=fs, output='sos')
        x = 0.5 * signal.sosfilt(sos, noise)
        
    elif signal_type == "Triple tono":
        # Three sinusoids for DSB-LC example
        x = (0.5 * np.sin(2 * np.pi * 500 * t) + 
             0.3 * np.sin(2 * np.pi * 1000 * t) + 
             0.2 * np.sin(2 * np.pi * 1500 * t))
    
    else:
        # Default: simple tone
        x = 0.8 * np.sin(2 * np.pi * 1000 * t)
    
    # Normalize to prevent clipping
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x)) * 0.9
    
    return x, fs

def apply_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Apply digital filter with zero-phase delay compensation.
    
    Args:
        x: Input signal
        sos: Second-order sections filter coefficients
        
    Returns:
        Filtered signal
        
    Note:
        Uses filtfilt for zero-phase filtering to avoid group delay.
    """
    return signal.sosfiltfilt(sos, x)

def compute_crosstalk(signal_desired: np.ndarray, signal_received: np.ndarray) -> float:
    """
    Compute crosstalk between desired and received signals.
    
    Args:
        signal_desired: Original signal that should be isolated
        signal_received: Actually received signal (may contain leakage)
        
    Returns:
        Crosstalk in dB (negative values indicate good isolation)
        
    Mathematical Note:
        Crosstalk = 20*log10(|unwanted_component| / |desired_component|)
    """
    # Ensure same length
    min_len = min(len(signal_desired), len(signal_received))
    s1 = signal_desired[:min_len]
    s2 = signal_received[:min_len]
    
    # Power calculations
    power_desired = np.mean(s1**2)
    power_received = np.mean(s2**2)
    
    if power_desired == 0 or power_received == 0:
        return -np.inf
    
    # Cross-correlation to find leakage
    correlation = np.corrcoef(s1, s2)[0, 1]
    leakage_power = correlation**2 * power_received
    
    if leakage_power == 0:
        return -np.inf
    
    crosstalk_db = 10 * np.log10(leakage_power / power_desired)
    
    return crosstalk_db

def phase_error_analysis(xI_orig: np.ndarray, xQ_orig: np.ndarray, 
                        xI_recv: np.ndarray, xQ_recv: np.ndarray) -> dict:
    """
    Analyze phase error effects in I/Q demodulation.
    
    Args:
        xI_orig, xQ_orig: Original I and Q signals
        xI_recv, xQ_recv: Received I and Q signals
        
    Returns:
        Dictionary with phase error metrics
    """
    # Compute cross-correlations
    corr_II = np.corrcoef(xI_orig, xI_recv)[0, 1]
    corr_QQ = np.corrcoef(xQ_orig, xQ_recv)[0, 1]
    corr_IQ = np.corrcoef(xI_orig, xQ_recv)[0, 1]
    corr_QI = np.corrcoef(xQ_orig, xI_recv)[0, 1]
    
    # Estimate phase error
    phase_error_est = np.arctan2(corr_IQ + corr_QI, corr_II + corr_QQ)
    
    return {
        "phase_error_deg": np.rad2deg(phase_error_est),
        "I_to_I_correlation": corr_II,
        "Q_to_Q_correlation": corr_QQ, 
        "I_to_Q_leakage": corr_IQ,
        "Q_to_I_leakage": corr_QI,
        "isolation_dB": 20 * np.log10(max(abs(corr_IQ), abs(corr_QI)) / max(corr_II, corr_QQ))
    }