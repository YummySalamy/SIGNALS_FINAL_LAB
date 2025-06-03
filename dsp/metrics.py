"""
Signal Quality Metrics Module

Implements various metrics for evaluating signal processing performance
including SNR, THD, EVM, and spectral analysis tools.

Mathematical Foundations:
    SNR = 10*log₁₀(P_signal / P_noise) [dB]
    THD = 10*log₁₀(P_harmonics / P_fundamental) [dB]
    EVM = RMS(error_vector) / RMS(reference) * 100 [%]
"""

import numpy as np
from scipy import signal, stats
from typing import Tuple, Dict, Optional, Union
import warnings

def compute_snr(signal_clean: np.ndarray, signal_noisy: np.ndarray, 
                method: str = 'power') -> float:
    """
    Compute Signal-to-Noise Ratio between clean and noisy signals.
    
    Args:
        signal_clean: Original clean signal
        signal_noisy: Signal with noise (or recovered signal)
        method: 'power' or 'amplitude' for SNR calculation
        
    Returns:
        SNR in dB
        
    Mathematical Note:
        Power SNR: 10*log₁₀(σ²_signal / σ²_noise)
        Amplitude SNR: 20*log₁₀(RMS_signal / RMS_noise)
    """
    # Ensure same length
    min_len = min(len(signal_clean), len(signal_noisy))
    s_clean = signal_clean[:min_len]
    s_noisy = signal_noisy[:min_len]
    
    # Compute noise as difference
    noise = s_noisy - s_clean
    
    # Signal power
    if method == 'power':
        power_signal = np.var(s_clean)  # Variance = AC power
        power_noise = np.var(noise)
        
        if power_noise == 0:
            return np.inf
        
        snr_db = 10 * np.log10(power_signal / power_noise)
        
    elif method == 'amplitude':
        rms_signal = np.sqrt(np.mean(s_clean**2))
        rms_noise = np.sqrt(np.mean(noise**2))
        
        if rms_noise == 0:
            return np.inf
        
        snr_db = 20 * np.log10(rms_signal / rms_noise)
        
    else:
        raise ValueError(f"Unknown SNR method: {method}")
    
    return snr_db

def compute_sinad(signal: np.ndarray, fs: float, 
                  fundamental_freq: float) -> Tuple[float, float, float]:
    """
    Compute Signal-to-Noise-And-Distortion ratio.
    
    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        fundamental_freq: Expected fundamental frequency (Hz)
        
    Returns:
        Tuple of (SINAD_dB, SNR_dB, THD_dB)
        
    Mathematical Note:
        SINAD considers both noise and harmonic distortion.
        THD includes harmonic content up to Nyquist frequency.
    """
    # Compute power spectral density
    f, Pxx = signal.welch(signal, fs, nperseg=len(signal)//4)
    
    # Find fundamental frequency bin
    fund_idx = np.argmin(np.abs(f - fundamental_freq))
    fund_power = Pxx[fund_idx]
    
    # Find harmonics (2f, 3f, 4f, ...)
    harmonic_power = 0
    max_harmonic = int(fs/2 / fundamental_freq)
    
    for h in range(2, max_harmonic + 1):
        harm_freq = h * fundamental_freq
        if harm_freq < fs/2:
            harm_idx = np.argmin(np.abs(f - harm_freq))
            # Sum power in ±3 bins around harmonic
            start_idx = max(0, harm_idx - 3)
            end_idx = min(len(Pxx), harm_idx + 4)
            harmonic_power += np.sum(Pxx[start_idx:end_idx])
    
    # Total power excluding DC
    total_power = np.sum(Pxx[1:])  # Skip DC bin
    
    # Noise power = total - fundamental - harmonics
    noise_power = total_power - fund_power - harmonic_power
    
    # Compute metrics
    if fund_power > 0:
        snr_db = 10 * np.log10(fund_power / noise_power) if noise_power > 0 else np.inf
        thd_db = 10 * np.log10(harmonic_power / fund_power) if harmonic_power > 0 else -np.inf
        sinad_db = 10 * np.log10(fund_power / (noise_power + harmonic_power))
    else:
        snr_db = thd_db = sinad_db = -np.inf
    
    return sinad_db, snr_db, thd_db

def compute_psd(input_signal: np.ndarray, fs: float, 
                method: str = 'welch', 
                nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density with normalization.
    
    Args:
        input_signal: Input signal (renamed to avoid conflict with scipy.signal)
        fs: Sampling frequency (Hz)
        method: 'welch', 'periodogram', or 'multitaper'
        nperseg: Length of each segment for Welch method
        
    Returns:
        Tuple of (frequencies, PSD_dB)
        
    Mathematical Note:
        PSD normalized to 0 dB at peak for visualization clarity.
        Units: dB relative to peak power.
    """
    if len(input_signal) == 0:
        return np.array([]), np.array([])
    
    if nperseg is None:
        nperseg = min(len(input_signal)//4, 2048)
    
    if method == 'welch':
        f, Pxx = signal.welch(input_signal, fs, nperseg=nperseg, 
                             window='hamming', noverlap=nperseg//2)
    elif method == 'periodogram':
        f, Pxx = signal.periodogram(input_signal, fs, window='hamming')
    elif method == 'multitaper':
        f, Pxx = signal.multitaper(input_signal, fs, NW=4)
    else:
        raise ValueError(f"Unknown PSD method: {method}")
    
    # Convert to dB and normalize to peak
    Pxx_db = 10 * np.log10(Pxx + 1e-12)  # Avoid log(0)
    Pxx_db_norm = Pxx_db - np.max(Pxx_db)  # Normalize to 0 dB peak
    
    return f, Pxx_db_norm

def compute_evm(reference: np.ndarray, measured: np.ndarray) -> float:
    """
    Compute Error Vector Magnitude for modulation quality.
    
    Args:
        reference: Ideal signal constellation points
        measured: Actual received signal points
        
    Returns:
        EVM percentage
        
    Mathematical Note:
        EVM = 100 * RMS(error_vector) / RMS(reference_vector)
        Lower EVM indicates better modulation quality.
    """
    # Ensure same length
    min_len = min(len(reference), len(measured))
    ref = reference[:min_len]
    meas = measured[:min_len]
    
    # Error vector
    error = meas - ref
    
    # RMS calculations
    rms_error = np.sqrt(np.mean(np.abs(error)**2))
    rms_reference = np.sqrt(np.mean(np.abs(ref)**2))
    
    if rms_reference == 0:
        return np.inf
    
    evm_percent = 100 * rms_error / rms_reference
    
    return evm_percent

def compute_papr(signal: np.ndarray) -> float:
    """
    Compute Peak-to-Average Power Ratio.
    
    Args:
        signal: Input signal
        
    Returns:
        PAPR in dB
        
    Mathematical Note:
        PAPR = 10*log₁₀(P_peak / P_average)
        High PAPR indicates signal with large dynamic range.
    """
    if len(signal) == 0:
        return 0.0
    
    peak_power = np.max(signal**2)
    avg_power = np.mean(signal**2)
    
    if avg_power == 0:
        return np.inf
    
    papr_db = 10 * np.log10(peak_power / avg_power)
    
    return papr_db

def compute_spectral_efficiency(signal: np.ndarray, fs: float, 
                               bandwidth: float) -> float:
    """
    Estimate spectral efficiency in bits/s/Hz.
    
    Args:
        signal: Modulated signal
        fs: Sampling frequency (Hz)
        bandwidth: Signal bandwidth (Hz)
        
    Returns:
        Spectral efficiency estimate
        
    Note:
        This is a simplified estimate based on signal entropy.
    """
    # Estimate signal entropy (information content)
    hist, _ = np.histogram(signal, bins=256, density=True)
    hist = hist[hist > 0]  # Remove zero bins
    entropy = -np.sum(hist * np.log2(hist))
    
    # Normalize by bandwidth
    if bandwidth > 0:
        spectral_eff = entropy * fs / bandwidth / 2  # Factor of 2 for complex signals
    else:
        spectral_eff = 0.0
    
    return spectral_eff

def analyze_phase_noise(signal: np.ndarray, fs: float) -> Dict:
    """
    Analyze phase noise characteristics of a signal.
    
    Args:
        signal: Complex-valued signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Dictionary with phase noise metrics
    """
    if np.iscomplexobj(signal):
        phase = np.angle(signal)
    else:
        # Use Hilbert transform to get analytic signal
        analytic = signal.hilbert(signal)
        phase = np.angle(analytic)
    
    # Unwrap phase to avoid 2π jumps
    phase_unwrapped = np.unwrap(phase)
    
    # Detrend to remove linear phase (frequency offset)
    phase_detrended = signal.detrend(phase_unwrapped)
    
    # Phase noise PSD
    f_phase, Sφ = signal.welch(phase_detrended, fs, nperseg=len(phase_detrended)//4)
    
    # Convert to dBc/Hz (dB relative to carrier per Hz)
    Sφ_dBc = 10 * np.log10(Sφ + 1e-12) - 10 * np.log10(fs/len(phase_detrended))
    
    # RMS phase jitter (integrated phase noise)
    phase_jitter_rms = np.sqrt(np.var(phase_detrended))
    
    return {
        'phase_noise_psd_dBc': Sφ_dBc,
        'frequencies': f_phase,
        'rms_phase_jitter_rad': phase_jitter_rms,
        'rms_phase_jitter_deg': np.rad2deg(phase_jitter_rms)
    }

def frequency_offset_estimate(signal: np.ndarray, fs: float, 
                             method: str = 'fft') -> float:
    """
    Estimate frequency offset in a signal.
    
    Args:
        signal: Input signal (real or complex)
        fs: Sampling frequency (Hz)
        method: 'fft' or 'autocorr'
        
    Returns:
        Estimated frequency offset in Hz
    """
    if method == 'fft':
        # Find peak in spectrum
        f, Pxx = compute_psd(signal, fs, method='periodogram')
        peak_idx = np.argmax(Pxx)
        freq_offset = f[peak_idx]
        
        # Correct for DC-centered spectrum
        if freq_offset > fs/2:
            freq_offset -= fs
            
    elif method == 'autocorr':
        # Autocorrelation method for periodic signals
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after delay=0
        peaks, _ = signal.find_peaks(autocorr[1:])
        if len(peaks) > 0:
            period_samples = peaks[0] + 1
            freq_offset = fs / period_samples
        else:
            freq_offset = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return freq_offset

def compute_ber_estimate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Estimate Bit Error Rate between transmitted and received bit streams.
    
    Args:
        tx_bits: Transmitted bits (0s and 1s)
        rx_bits: Received bits (0s and 1s)
        
    Returns:
        BER (fraction of errors)
    """
    # Ensure same length
    min_len = min(len(tx_bits), len(rx_bits))
    tx = tx_bits[:min_len].astype(int)
    rx = rx_bits[:min_len].astype(int)
    
    # Count bit errors
    errors = np.sum(tx != rx)
    
    if min_len == 0:
        return 0.0
    
    ber = errors / min_len
    
    return ber

def constellation_analysis(signal: np.ndarray, 
                          modulation_type: str = 'qam') -> Dict:
    """
    Analyze constellation diagram metrics.
    
    Args:
        signal: Complex-valued constellation points
        modulation_type: 'qam', 'psk', or 'ask'
        
    Returns:
        Dictionary with constellation metrics
    """
    if not np.iscomplexobj(signal):
        # Convert to complex if real
        signal = signal.astype(complex)
    
    # Basic constellation metrics
    avg_power = np.mean(np.abs(signal)**2)
    peak_power = np.max(np.abs(signal)**2)
    
    # Magnitude and phase
    magnitude = np.abs(signal)
    phase = np.angle(signal)
    
    # Constellation specific analysis
    if modulation_type.lower() == 'qam':
        # I/Q components
        I = np.real(signal)
        Q = np.imag(signal)
        
        metrics = {
            'avg_power': avg_power,
            'peak_power': peak_power,
            'papr_db': 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0,
            'I_variance': np.var(I),
            'Q_variance': np.var(Q),
            'IQ_correlation': np.corrcoef(I, Q)[0, 1] if len(I) > 1 else 0,
            'magnitude_std': np.std(magnitude),
            'phase_std': np.std(phase)
        }
        
    elif modulation_type.lower() == 'psk':
        # Phase-based metrics
        phase_unwrapped = np.unwrap(phase)
        
        metrics = {
            'avg_power': avg_power,
            'magnitude_variation': np.std(magnitude) / np.mean(magnitude) if np.mean(magnitude) > 0 else 0,
            'phase_noise_std': np.std(np.diff(phase_unwrapped)),
            'phase_range': np.ptp(phase)
        }
        
    else:
        # Generic metrics
        metrics = {
            'avg_power': avg_power,
            'peak_power': peak_power,
            'magnitude_std': np.std(magnitude),
            'phase_std': np.std(phase)
        }
    
    return metrics

def signal_statistics(signal: np.ndarray) -> Dict:
    """
    Compute comprehensive signal statistics.
    
    Args:
        signal: Input signal (real or complex)
        
    Returns:
        Dictionary with statistical measures
    """
    if np.iscomplexobj(signal):
        # Complex signal statistics
        magnitude = np.abs(signal)
        phase = np.angle(signal)
        
        stats_dict = {
            'length': len(signal),
            'magnitude_mean': np.mean(magnitude),
            'magnitude_std': np.std(magnitude),
            'magnitude_min': np.min(magnitude),
            'magnitude_max': np.max(magnitude),
            'phase_mean': np.mean(phase),
            'phase_std': np.std(phase),
            'real_mean': np.mean(np.real(signal)),
            'real_std': np.std(np.real(signal)),
            'imag_mean': np.mean(np.imag(signal)),
            'imag_std': np.std(np.imag(signal)),
            'power': np.mean(magnitude**2)
        }
    else:
        # Real signal statistics
        stats_dict = {
            'length': len(signal),
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'rms': np.sqrt(np.mean(signal**2)),
            'peak_to_peak': np.ptp(signal),
            'skewness': stats.skew(signal),
            'kurtosis': stats.kurtosis(signal),
            'power': np.mean(signal**2)
        }
    
    return stats_dict

def dynamic_range_analysis(signal: np.ndarray, fs: float) -> Dict:
    """
    Analyze dynamic range characteristics.
    
    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Dictionary with dynamic range metrics
    """
    # Power spectral density
    f, Pxx_db = compute_psd(signal, fs)
    
    # Find peak and noise floor
    peak_power_db = np.max(Pxx_db)
    
    # Estimate noise floor (bottom 10% of spectrum)
    sorted_pxx = np.sort(Pxx_db)
    noise_floor_db = np.mean(sorted_pxx[:len(sorted_pxx)//10])
    
    # Spurious-free dynamic range
    sfdr_db = peak_power_db - noise_floor_db
    
    # Signal bandwidth (where power is within 3 dB of peak)
    bw_mask = Pxx_db >= (peak_power_db - 3)
    if np.any(bw_mask):
        bw_indices = np.where(bw_mask)[0]
        signal_bandwidth = f[bw_indices[-1]] - f[bw_indices[0]]
    else:
        signal_bandwidth = 0.0
    
    return {
        'peak_power_db': peak_power_db,
        'noise_floor_db': noise_floor_db,
        'sfdr_db': sfdr_db,
        'signal_bandwidth_hz': signal_bandwidth,
        'frequency_resolution_hz': f[1] - f[0] if len(f) > 1 else 0
    }