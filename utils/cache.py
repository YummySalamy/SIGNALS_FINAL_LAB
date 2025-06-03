"""
Caching Utilities for Streamlit

Provides optimized caching decorators for expensive computations
in signal processing applications.

Features:
- Automatic cache management
- Memory-efficient storage
- Hash optimization for arrays
- Resource caching for filters
"""

import streamlit as st
import numpy as np
import hashlib
from typing import Any, Callable, Union, Tuple
import pickle

def hash_array(arr: np.ndarray) -> str:
    """
    Create hash for numpy array for caching.
    
    Args:
        arr: Numpy array to hash
        
    Returns:
        Hash string
    """
    if arr.size == 0:
        return "empty_array"
    
    # Use array bytes for hashing (more reliable than array itself)
    array_bytes = arr.tobytes()
    hash_obj = hashlib.md5(array_bytes + str(arr.shape).encode() + str(arr.dtype).encode())
    return hash_obj.hexdigest()

def hash_params(**kwargs) -> str:
    """
    Create hash for function parameters.
    
    Args:
        **kwargs: Parameters to hash
        
    Returns:
        Hash string
    """
    # Convert parameters to string and hash
    param_str = str(sorted(kwargs.items()))
    hash_obj = hashlib.md5(param_str.encode())
    return hash_obj.hexdigest()

@st.cache_data(show_spinner=False)
def cached_fourier_coefficients(example_id: str, N: int, _cache_key: str = None) -> dict:
    """
    Cached computation of Fourier coefficients.
    
    Args:
        example_id: Signal identifier
        N: Number of harmonics
        _cache_key: Optional cache key (not used in computation)
        
    Returns:
        Dictionary with Fourier coefficients
    """
    from dsp.fourier import compute_fourier_coefficients
    return compute_fourier_coefficients(example_id, N)

@st.cache_data(show_spinner=False)
def cached_signal_reconstruction(coeffs_hash: str, t_hash: str, T: float, 
                               _coeffs: dict = None, _t: np.ndarray = None) -> np.ndarray:
    """
    Cached signal reconstruction from Fourier coefficients.
    
    Args:
        coeffs_hash: Hash of coefficients dict
        t_hash: Hash of time vector
        T: Period
        _coeffs: Actual coefficients (not used for caching)
        _t: Actual time vector (not used for caching)
        
    Returns:
        Reconstructed signal
    """
    from dsp.fourier import reconstruct_fourier_series
    return reconstruct_fourier_series(_coeffs, _t, T)

@st.cache_data(show_spinner=False)
def cached_psd_computation(signal_hash: str, fs: float, method: str = 'welch',
                          _signal: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached power spectral density computation.
    
    Args:
        signal_hash: Hash of input signal
        fs: Sampling frequency
        method: PSD method
        _signal: Actual signal (not used for caching)
        
    Returns:
        Tuple of (frequencies, PSD)
    """
    from dsp.metrics import compute_psd
    return compute_psd(_signal, fs, method)

@st.cache_data(show_spinner=False)
def cached_modulation_demodulation(signal_hash: str, fc: float, fs: float, 
                                  filter_order: int, mod_type: str = 'dsb_sc',
                                  _signal: np.ndarray = None) -> dict:
    """
    Cached modulation and demodulation processing.
    
    Args:
        signal_hash: Hash of input signal
        fc: Carrier frequency
        fs: Sampling frequency
        filter_order: Filter order
        mod_type: Modulation type
        _signal: Actual signal (not used for caching)
        
    Returns:
        Dictionary with modulated and demodulated signals
    """
    from dsp.modulation import dsb_sc_modulate, dsb_sc_demodulate
    from dsp.filters import design_lowpass_filter
    
    # Modulation
    modulated = dsb_sc_modulate(_signal, fc, fs)
    
    # Design filter
    sos = design_lowpass_filter(0.8 * fc, fs, filter_order)
    
    # Demodulation
    demodulated = dsb_sc_demodulate(modulated, fc, fs, sos)
    
    return {
        'modulated': modulated,
        'demodulated': demodulated,
        'filter_sos': sos
    }

@st.cache_resource
def cached_filter_design(cutoff: float, fs: float, order: int, 
                        filter_type: str = 'fir', window: str = 'hamming') -> np.ndarray:
    """
    Cached filter design (uses cache_resource for filter objects).
    
    Args:
        cutoff: Cutoff frequency
        fs: Sampling frequency
        order: Filter order
        filter_type: Type of filter
        window: Window function
        
    Returns:
        Filter coefficients (SOS or taps)
    """
    from dsp.filters import design_lowpass_filter
    return design_lowpass_filter(cutoff, fs, order, filter_type, window)

@st.cache_data(show_spinner=False)
def cached_demo_signal(signal_type: str, duration: float, fs: float) -> Tuple[np.ndarray, float]:
    """
    Cached demo signal generation.
    
    Args:
        signal_type: Type of demo signal
        duration: Duration in seconds
        fs: Sampling frequency
        
    Returns:
        Tuple of (signal, actual_fs)
    """
    from dsp.modulation import generate_demo_signal
    return generate_demo_signal(signal_type, duration, fs)

@st.cache_data(show_spinner=False)
def cached_metrics_computation(signal1_hash: str, signal2_hash: str,
                              _signal1: np.ndarray = None, 
                              _signal2: np.ndarray = None) -> dict:
    """
    Cached computation of signal quality metrics.
    
    Args:
        signal1_hash: Hash of first signal
        signal2_hash: Hash of second signal
        _signal1: First signal (not used for caching)
        _signal2: Second signal (not used for caching)
        
    Returns:
        Dictionary with computed metrics
    """
    from dsp.metrics import compute_snr, compute_evm, signal_statistics
    
    # Ensure same length
    min_len = min(len(_signal1), len(_signal2))
    s1 = _signal1[:min_len]
    s2 = _signal2[:min_len]
    
    metrics = {
        'snr': compute_snr(s1, s2),
        'correlation': np.corrcoef(s1, s2)[0, 1] if len(s1) > 1 else 0,
        'rms_error': np.sqrt(np.mean((s1 - s2)**2)),
        'stats_1': signal_statistics(s1),
        'stats_2': signal_statistics(s2)
    }
    
    return metrics

def create_cache_key(*args, **kwargs) -> str:
    """
    Create cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    key_components = []
    
    # Handle positional arguments
    for arg in args:
        if isinstance(arg, np.ndarray):
            key_components.append(hash_array(arg))
        else:
            key_components.append(str(arg))
    
    # Handle keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, np.ndarray):
            key_components.append(f"{key}:{hash_array(value)}")
        else:
            key_components.append(f"{key}:{value}")
    
    # Combine and hash
    combined = "_".join(key_components)
    return hashlib.md5(combined.encode()).hexdigest()

def cached_computation(func: Callable) -> Callable:
    """
    Decorator for caching expensive computations.
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function with caching
    """
    @st.cache_data(show_spinner=False)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper

def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("🗑️ All caches cleared!")

def get_cache_stats() -> dict:
    """
    Get cache statistics (if available).
    
    Returns:
        Dictionary with cache statistics
    """
    # Note: Streamlit doesn't provide direct access to cache stats
    # This is a placeholder for future implementation
    return {
        'data_cache_entries': 'N/A',
        'resource_cache_entries': 'N/A',
        'total_memory_usage': 'N/A'
    }

def cache_management_panel():
    """Create cache management panel in sidebar."""
    with st.sidebar.expander("🗄️ Cache Management"):
        st.markdown("**Cache Status:**")
        
        # Display cache stats
        stats = get_cache_stats()
        for key, value in stats.items():
            st.text(f"{key}: {value}")
        
        # Clear cache button
        if st.button("🗑️ Clear All Caches"):
            clear_all_caches()
        
        # Cache settings
        st.markdown("**Cache Settings:**")
        
        # Enable/disable caching
        use_cache = st.checkbox("Enable caching", value=True)
        if not use_cache:
            st.warning("⚠️ Disabling cache will slow down computations")
        
        # Memory limit
        memory_limit = st.slider(
            "Memory limit (MB):",
            min_value=100,
            max_value=2000,
            value=500,
            help="Maximum memory for cached data"
        )
        
        return use_cache, memory_limit

# Advanced caching utilities for specific use cases

@st.cache_data(show_spinner=False, max_entries=10)
def cached_filter_response(filter_hash: str, fs: float, N: int = 8192,
                          _filter_coeffs: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached filter frequency response computation.
    
    Args:
        filter_hash: Hash of filter coefficients
        fs: Sampling frequency
        N: Number of frequency points
        _filter_coeffs: Actual filter coefficients
        
    Returns:
        Tuple of (frequencies, magnitude, phase)
    """
    from dsp.filters import get_filter_response
    return get_filter_response(_filter_coeffs, fs, N)

@st.cache_data(show_spinner=False, max_entries=5)
def cached_convergence_analysis(example_id: str, N_max: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached Fourier series convergence analysis.
    
    Args:
        example_id: Signal identifier
        N_max: Maximum number of harmonics
        
    Returns:
        Tuple of (N_values, errors)
    """
    from dsp.fourier import get_convergence_analysis
    return get_convergence_analysis(example_id, N_max)

@st.cache_data(show_spinner=False)
def cached_iq_processing(xI_hash: str, xQ_hash: str, fc: float, fs: float,
                        filter_order: int, phase_error: float = 0.0,
                        _xI: np.ndarray = None, _xQ: np.ndarray = None) -> dict:
    """
    Cached I/Q modulation and demodulation.
    
    Args:
        xI_hash: Hash of I signal
        xQ_hash: Hash of Q signal
        fc: Carrier frequency
        fs: Sampling frequency
        filter_order: Filter order
        phase_error: Phase error in radians
        _xI: I signal (not used for caching)
        _xQ: Q signal (not used for caching)
        
    Returns:
        Dictionary with I/Q processing results
    """
    from dsp.modulation import iq_modulate, iq_demodulate
    from dsp.filters import design_lowpass_filter
    
    # Modulation
    s_tx = iq_modulate(_xI, _xQ, fc, fs)
    
    # Design filter
    sos = design_lowpass_filter(0.6 * fc, fs, filter_order)
    
    # Demodulation
    xI_rx, xQ_rx = iq_demodulate(s_tx, fc, fs, sos, phase_error)
    
    return {
        'transmitted': s_tx,
        'xI_recovered': xI_rx,
        'xQ_recovered': xQ_rx,
        'filter_sos': sos
    }

# Context manager for temporary cache disable
class disable_cache:
    """Context manager to temporarily disable caching."""
    
    def __enter__(self):
        # Store original cache state
        self._cache_enabled = True  # Placeholder
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore cache state
        pass

# Utility function to estimate memory usage
def estimate_array_memory(arr: np.ndarray) -> float:
    """
    Estimate memory usage of numpy array in MB.
    
    Args:
        arr: Numpy array
        
    Returns:
        Memory usage in MB
    """
    bytes_per_element = arr.dtype.itemsize
    total_bytes = arr.size * bytes_per_element
    return total_bytes / (1024 * 1024)

def cache_size_warning(arrays: list, threshold_mb: float = 100.0) -> None:
    """
    Check if arrays exceed memory threshold and warn user.
    
    Args:
        arrays: List of numpy arrays to check
        threshold_mb: Memory threshold in MB
    """
    total_memory = sum(estimate_array_memory(arr) for arr in arrays)
    
    if total_memory > threshold_mb:
        st.warning(f"⚠️ Large data detected ({total_memory:.1f} MB). "
                  f"Processing may be slow. Consider reducing signal length.")

def optimize_cache_usage():
    """Optimize cache usage by clearing old entries."""
    # This would implement cache optimization logic
    # For now, it's a placeholder
    pass