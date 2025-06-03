"""
Digital Filter Design Module

Implements various digital filters for signal processing applications,
specifically optimized for communication systems and demodulation.

Filter Types:
1. FIR filters using window methods (Hamming, Kaiser)
2. IIR filters (Butterworth, Chebyshev)  
3. Specialized filters for audio and RF applications

Mathematical Foundation:
    FIR: y[n] = Σ h[k]x[n-k]
    IIR: H(z) = B(z)/A(z) in second-order sections (SOS) form
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple, Optional
import warnings

def design_lowpass_filter(cutoff_freq: float, fs: float, order: int, 
                         filter_type: str = 'fir', window: str = 'hamming') -> np.ndarray:
    """
    Design a low-pass filter for demodulation applications.
    
    Args:
        cutoff_freq: Cutoff frequency in Hz
        fs: Sampling frequency in Hz  
        order: Filter order (number of taps for FIR)
        filter_type: 'fir' or 'iir'
        window: Window function for FIR design
        
    Returns:
        Filter coefficients (SOS format for IIR, taps for FIR)
        
    Mathematical Note:
        For communication systems, FIR filters provide linear phase
        but require higher orders. IIR filters are more efficient
        but introduce group delay.
    """
    # Validate inputs
    if cutoff_freq >= fs/2:
        warnings.warn(f"Cutoff {cutoff_freq} Hz exceeds Nyquist {fs/2} Hz")
        cutoff_freq = 0.8 * fs/2
    
    if cutoff_freq <= 0:
        raise ValueError("Cutoff frequency must be positive")
    
    # Normalized frequency
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    if filter_type.lower() == 'fir':
        # FIR filter using windowing method
        if order % 2 == 0:
            order += 1  # Ensure odd order for symmetric FIR
            
        if window.lower() == 'hamming':
            taps = signal.firwin(order, normalized_cutoff, window='hamming')
        elif window.lower() == 'kaiser':
            # Kaiser window with beta for good stopband attenuation
            beta = 8.0  # Provides ~60 dB stopband attenuation
            taps = signal.firwin(order, normalized_cutoff, window=('kaiser', beta))
        elif window.lower() == 'blackman':
            taps = signal.firwin(order, normalized_cutoff, window='blackman')
        else:
            taps = signal.firwin(order, normalized_cutoff, window='hamming')
        
        # Convert to SOS format for consistent interface
        # For FIR, create SOS with only numerator
        sos = np.zeros((1, 6))
        if len(taps) <= 6:
            sos[0, :len(taps)] = taps
            sos[0, 3] = 1.0  # Denominator [1, 0, 0]
        else:
            # For long FIR, return as-is and handle in apply_filter
            return taps
            
        return sos
        
    elif filter_type.lower() == 'iir':
        # Limit IIR order to prevent overflow
        max_iir_order = 20  # Safe maximum for Butterworth
        if order > max_iir_order:
            warnings.warn(f"IIR order {order} too high, limiting to {max_iir_order}")
            order = max_iir_order
        
        # Additional safety check for normalized cutoff
        if normalized_cutoff >= 0.95:
            normalized_cutoff = 0.95
            warnings.warn("Normalized cutoff too high, limiting to 0.95")
        
        try:
            # IIR Butterworth filter (maximally flat passband)
            sos = signal.butter(order, normalized_cutoff, btype='low', 
                               analog=False, output='sos')
            return sos
        except (OverflowError, ValueError) as e:
            # Fallback to FIR if IIR fails
            warnings.warn(f"IIR design failed ({e}), falling back to FIR")
            # Use equivalent FIR filter
            fir_order = order * 20  # Rule of thumb: FIR needs ~20x more taps
            if fir_order % 2 == 0:
                fir_order += 1
            taps = signal.firwin(fir_order, normalized_cutoff, window='hamming')
            return taps
        
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

def design_bandpass_filter(f_low: float, f_high: float, fs: float, 
                          order: int = 6) -> np.ndarray:
    """
    Design a bandpass filter for RF applications.
    
    Args:
        f_low: Lower cutoff frequency (Hz)
        f_high: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        SOS coefficients for bandpass filter
    """
    nyquist = fs / 2
    
    if f_low >= f_high:
        raise ValueError("Lower cutoff must be less than upper cutoff")
    
    if f_high >= nyquist:
        warnings.warn(f"Upper cutoff {f_high} Hz exceeds Nyquist {nyquist} Hz")
        f_high = 0.95 * nyquist
    
    # Normalized frequencies  
    low_norm = f_low / nyquist
    high_norm = f_high / nyquist
    
    sos = signal.butter(order, [low_norm, high_norm], btype='band', 
                       analog=False, output='sos')
    
    return sos

def design_hilbert_filter(order: int = 101) -> np.ndarray:
    """
    Design a Hilbert transform filter for envelope detection.
    
    Args:
        order: Filter order (should be odd)
        
    Returns:
        FIR filter coefficients for 90-degree phase shift
        
    Mathematical Note:
        Hilbert transform provides 90° phase shift for all frequencies.
        Used in envelope detection: |x + jH{x}|
    """
    if order % 2 == 0:
        order += 1  # Ensure odd order
    
    # Create Hilbert FIR filter
    taps = signal.firwin(order, 1.0, window='hamming', pass_zero=False)
    
    return taps

def apply_fir_filter(x: np.ndarray, taps: np.ndarray, 
                    zero_phase: bool = True) -> np.ndarray:
    """
    Apply FIR filter with optional zero-phase filtering.
    
    Args:
        x: Input signal
        taps: FIR filter coefficients
        zero_phase: Use filtfilt for zero-phase delay
        
    Returns:
        Filtered signal
    """
    if zero_phase:
        # Zero-phase filtering (no group delay)
        return signal.filtfilt(taps, 1.0, x)
    else:
        # Standard filtering (with group delay)
        return signal.lfilter(taps, 1.0, x)

def apply_iir_filter(x: np.ndarray, sos: np.ndarray, 
                    zero_phase: bool = False) -> np.ndarray:
    """
    Apply IIR filter in SOS form.
    
    Args:
        x: Input signal
        sos: Second-order sections coefficients or FIR taps
        zero_phase: Use sosfiltfilt for zero-phase delay
        
    Returns:
        Filtered signal
        
    Note:
        Zero-phase IIR filtering doubles the filter order.
        This function also handles FIR taps for compatibility.
    """
    # Check if it's actually FIR taps (1D array)
    if sos.ndim == 1:
        # It's FIR taps, use FIR filtering
        if zero_phase:
            return signal.filtfilt(sos, 1.0, x)
        else:
            return signal.lfilter(sos, 1.0, x)
    else:
        # It's SOS format for IIR
        if zero_phase:
            return signal.sosfiltfilt(sos, x)
        else:
            return signal.sosfilt(sos, x)

def get_filter_response(sos_or_taps: np.ndarray, fs: float, 
                       N: int = 8192) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute filter frequency response.
    
    Args:
        sos_or_taps: Filter coefficients (SOS or FIR taps)
        fs: Sampling frequency (Hz)
        N: Number of frequency points
        
    Returns:
        Tuple of (frequencies, magnitude_dB, phase_deg)
    """
    if sos_or_taps.ndim == 2:  # SOS format
        w, h = signal.sosfreqz(sos_or_taps, worN=N, fs=fs)
    else:  # FIR taps
        w, h = signal.freqz(sos_or_taps, worN=N, fs=fs)
    
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)  # Avoid log(0)
    phase_deg = np.angle(h, deg=True)
    
    return w, magnitude_db, phase_deg

def estimate_group_delay(sos_or_taps: np.ndarray, fs: float) -> float:
    """
    Estimate average group delay of filter.
    
    Args:
        sos_or_taps: Filter coefficients
        fs: Sampling frequency (Hz)
        
    Returns:
        Group delay in samples
    """
    if sos_or_taps.ndim == 2:  # SOS format
        w, gd = signal.group_delay((sos_or_taps, 1), fs=fs)
    else:  # FIR taps
        w, gd = signal.group_delay(sos_or_taps, fs=fs)
    
    # Return average group delay in passband (first 10% of Nyquist)
    passband_idx = w <= 0.1 * fs/2
    if np.any(passband_idx):
        return np.mean(gd[passband_idx])
    else:
        return np.mean(gd)

def design_anti_aliasing_filter(signal_bw: float, fs: float) -> np.ndarray:
    """
    Design anti-aliasing filter for ADC protection.
    
    Args:
        signal_bw: Signal bandwidth (Hz)
        fs: Sampling frequency (Hz)
        
    Returns:
        SOS coefficients for anti-aliasing filter
        
    Mathematical Note:
        Cutoff typically set to 0.8 * signal_bw with sharp rolloff
        to prevent aliasing while preserving signal content.
    """
    # Conservative cutoff at 80% of signal bandwidth
    cutoff = 0.8 * signal_bw
    
    # Ensure adequate oversampling
    if fs < 5 * signal_bw:
        warnings.warn(f"Low oversampling ratio: fs={fs}, signal_bw={signal_bw}")
    
    # 8th-order elliptic filter for sharp transition
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    
    sos = signal.ellip(8, 1, 60, normalized_cutoff, btype='low', 
                      analog=False, output='sos')
    
    return sos

def adaptive_filter_coeffs(input_signal: np.ndarray, desired_signal: np.ndarray,
                          order: int = 32, mu: float = 0.01) -> np.ndarray:
    """
    Simple LMS adaptive filter for noise cancellation.
    
    Args:
        input_signal: Reference input (noise source)
        desired_signal: Desired signal + noise
        order: Number of filter taps
        mu: Step size (learning rate)
        
    Returns:
        Optimized filter coefficients
        
    Mathematical Note:
        LMS algorithm: w[n+1] = w[n] + μe[n]x[n]
        where e[n] = d[n] - y[n] is the error signal
    """
    N = len(input_signal)
    w = np.zeros(order)  # Initialize filter weights
    y = np.zeros(N)      # Filter output
    
    for n in range(order, N):
        # Extract input vector
        x = input_signal[n-order:n][::-1]  # Reverse for convolution
        
        # Filter output
        y[n] = np.dot(w, x)
        
        # Error signal
        e = desired_signal[n] - y[n]
        
        # Update weights (LMS)
        w += mu * e * x
    
    return w

def design_reconstruction_filter(fs: float, oversampling_factor: int = 4) -> np.ndarray:
    """
    Design reconstruction filter for DAC output.
    
    Args:
        fs: Original sampling frequency (Hz)
        oversampling_factor: Upsampling ratio
        
    Returns:
        FIR filter coefficients for reconstruction
        
    Mathematical Note:
        Removes imaging artifacts after upsampling.
        Cutoff at fs/2 to preserve original signal bandwidth.
    """
    # Upsampled frequency
    fs_up = fs * oversampling_factor
    
    # Cutoff at original Nyquist frequency
    cutoff = fs / 2
    normalized_cutoff = cutoff / (fs_up / 2)
    
    # Long FIR filter for sharp transition
    order = 8 * oversampling_factor + 1  # Rule of thumb
    if order % 2 == 0:
        order += 1
    
    taps = signal.firwin(order, normalized_cutoff, window='kaiser', beta=8.0)
    
    # Scale for unity gain
    taps *= oversampling_factor
    
    return taps

def filter_bank_analysis(x: np.ndarray, fs: float, num_bands: int = 8) -> Tuple[list, np.ndarray]:
    """
    Decompose signal into frequency bands using filter bank.
    
    Args:
        x: Input signal
        fs: Sampling frequency (Hz)
        num_bands: Number of frequency bands
        
    Returns:
        Tuple of (band_signals, center_frequencies)
    """
    nyquist = fs / 2
    band_edges = np.linspace(0, nyquist, num_bands + 1)
    center_freqs = (band_edges[:-1] + band_edges[1:]) / 2
    
    band_signals = []
    
    for i in range(num_bands):
        f_low = band_edges[i]
        f_high = band_edges[i + 1]
        
        if f_low == 0:  # First band is low-pass
            sos = signal.butter(4, f_high/nyquist, btype='low', output='sos')
        elif f_high == nyquist:  # Last band is high-pass
            sos = signal.butter(4, f_low/nyquist, btype='high', output='sos')
        else:  # Middle bands are band-pass
            sos = signal.butter(4, [f_low/nyquist, f_high/nyquist], 
                              btype='band', output='sos')
        
        # Filter signal
        band_signal = signal.sosfilt(sos, x)
        band_signals.append(band_signal)
    
    return band_signals, center_freqs