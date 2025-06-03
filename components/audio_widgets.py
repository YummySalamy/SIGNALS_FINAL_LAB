"""
Audio Widget Components for Streamlit

Provides audio playback, file upload, and download functionality
optimized for signal processing applications.

Features:
- Audio file upload with format conversion
- Interactive audio player with waveform display
- Audio download as WAV files
- Real-time audio processing
"""

import streamlit as st
import numpy as np
import soundfile as sf
import io
import tempfile
import warnings
from typing import Tuple, Optional, Union
import plotly.graph_objects as go

def create_audio_player(signal: np.ndarray, fs: float, 
                       key: str = None, 
                       show_waveform: bool = True) -> None:
    """
    Create an interactive audio player with optional waveform display.
    
    Args:
        signal: Audio signal array
        fs: Sampling frequency (Hz)
        key: Unique key for Streamlit widget
        show_waveform: Whether to show waveform plot
    """
    # Ensure signal is in valid range [-1, 1]
    if np.max(np.abs(signal)) > 0:
        signal_normalized = signal / np.max(np.abs(signal)) * 0.95
    else:
        signal_normalized = signal
    
    # Convert to bytes for audio player
    audio_bytes = audio_to_wav_bytes(signal_normalized, fs)
    
    # Create audio player
    st.audio(audio_bytes, format='audio/wav', start_time=0)
    
    if show_waveform:
        # Display compact waveform
        duration = len(signal) / fs
        if duration > 10:  # For long signals, show first 10 seconds
            n_samples = int(10 * fs)
            t_display = np.arange(n_samples) / fs
            signal_display = signal_normalized[:n_samples]
        else:
            t_display = np.arange(len(signal_normalized)) / fs
            signal_display = signal_normalized
        
        # Simple waveform plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_display, y=signal_display,
            mode='lines',
            name='Waveform',
            line=dict(color='#007acc', width=1),
            hovertemplate='Time: %{x:.3f} s<br>Amplitude: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            height=150,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def audio_to_wav_bytes(signal: np.ndarray, fs: float) -> bytes:
    """
    Convert audio signal to WAV format bytes for download/playback.
    
    Args:
        signal: Audio signal array
        fs: Sampling frequency (Hz)
        
    Returns:
        WAV file as bytes
    """
    # Ensure 16-bit integer format for WAV
    if signal.dtype != np.int16:
        # Normalize and convert to 16-bit
        if np.max(np.abs(signal)) > 0:
            signal_norm = signal / np.max(np.abs(signal))
        else:
            signal_norm = signal
        signal_int16 = (signal_norm * 32767).astype(np.int16)
    else:
        signal_int16 = signal
    
    # Create WAV file in memory
    with io.BytesIO() as buffer:
        sf.write(buffer, signal_int16, int(fs), format='WAV', subtype='PCM_16')
        wav_bytes = buffer.getvalue()
    
    return wav_bytes

def load_audio_file(uploaded_file) -> Tuple[np.ndarray, float]:
    """
    Load and preprocess uploaded audio file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (signal, sample_rate)
    """
    if uploaded_file is None:
        return np.array([]), 0.0
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio file
        signal, fs = sf.read(tmp_file_path)
        
        # Convert stereo to mono if needed
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
            st.info("🔄 Converted stereo to mono")
        
        # Ensure float32 format
        signal = signal.astype(np.float32)
        
        return signal, fs
        
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return np.array([]), 0.0

def create_audio_uploader(label: str = "Upload audio file", 
                         key: str = None,
                         help_text: str = None) -> Optional[Tuple[np.ndarray, float]]:
    """
    Create audio file uploader with automatic processing.
    
    Args:
        label: Upload button label
        key: Unique key for widget
        help_text: Help text for user
        
    Returns:
        Tuple of (signal, fs) or None if no file uploaded
    """
    if help_text is None:
        help_text = "Supported formats: WAV, MP3, FLAC, OGG. Files will be converted to mono."
    
    uploaded_file = st.file_uploader(
        label,
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        key=key,
        help=help_text
    )
    
    if uploaded_file is not None:
        signal, fs = load_audio_file(uploaded_file)
        
        if len(signal) > 0:
            # Display file info
            duration = len(signal) / fs
            file_size = len(signal) * 4 / 1024 / 1024  # MB (float32)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{duration:.1f} s")
            with col2:
                st.metric("Sample Rate", f"{fs/1000:.1f} kHz")
            with col3:
                st.metric("File Size", f"{file_size:.1f} MB")
            
            return signal, fs
    
    return None

def create_download_button(signal: np.ndarray, fs: float, 
                          filename: str = "processed_audio.wav",
                          label: str = "💾 Download Audio") -> None:
    """
    Create download button for processed audio.
    
    Args:
        signal: Audio signal to download
        fs: Sampling frequency (Hz)
        filename: Output filename
        label: Button label
    """
    if len(signal) == 0:
        st.warning("No signal available for download")
        return
    
    # Convert to WAV bytes
    wav_bytes = audio_to_wav_bytes(signal, fs)
    
    # Create download button
    st.download_button(
        label=label,
        data=wav_bytes,
        file_name=filename,
        mime="audio/wav",
        help=f"Download {filename} ({len(wav_bytes)/1024:.1f} KB)"
    )

def audio_recorder_placeholder():
    """
    Placeholder for future audio recording functionality.
    """
    st.info("🎤 Audio recording feature coming soon!")
    st.markdown("""
    **Future features:**
    - Real-time audio recording from microphone
    - Live signal processing and analysis
    - Recording quality settings
    - Auto-save functionality
    """)

def create_audio_comparison_widget(original: np.ndarray, 
                                  processed: np.ndarray,
                                  fs: float,
                                  labels: Tuple[str, str] = ("Original", "Processed")) -> None:
    """
    Create side-by-side audio comparison widget.
    
    Args:
        original: Original audio signal
        processed: Processed audio signal
        fs: Sampling frequency (Hz)
        labels: Tuple of (original_label, processed_label)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{labels[0]}:**")
        create_audio_player(original, fs, key=f"audio_orig_{id(original)}", 
                          show_waveform=False)
        create_download_button(original, fs, 
                             f"{labels[0].lower().replace(' ', '_')}.wav",
                             f"⬇️ {labels[0]}")
    
    with col2:
        st.markdown(f"**{labels[1]}:**")
        create_audio_player(processed, fs, key=f"audio_proc_{id(processed)}", 
                          show_waveform=False)
        create_download_button(processed, fs,
                             f"{labels[1].lower().replace(' ', '_')}.wav", 
                             f"⬇️ {labels[1]}")

def validate_audio_parameters(signal: np.ndarray, fs: float, 
                             fc: float = None) -> bool:
    """
    Validate audio parameters for processing.
    
    Args:
        signal: Audio signal
        fs: Sampling frequency (Hz)
        fc: Carrier frequency (Hz, optional)
        
    Returns:
        True if parameters are valid
    """
    valid = True
    
    # Check signal properties
    if len(signal) == 0:
        st.error("❌ Empty signal")
        valid = False
    
    if fs <= 0:
        st.error("❌ Invalid sampling frequency")
        valid = False
    
    # Check for clipping
    if np.max(np.abs(signal)) >= 1.0:
        st.warning("⚠️ Signal may be clipped (peaks at ±1.0)")
    
    # Check dynamic range
    if np.std(signal) < 0.001:
        st.warning("⚠️ Very low signal amplitude")
    
    # Check carrier frequency vs Nyquist
    if fc is not None and fc >= fs/2:
        st.error(f"❌ Carrier frequency ({fc/1000:.1f} kHz) exceeds Nyquist limit ({fs/2000:.1f} kHz)")
        valid = False
    
    # Signal length recommendations
    duration = len(signal) / fs
    if duration < 1.0:
        st.warning("⚠️ Very short signal (< 1 second)")
    elif duration > 60.0:
        st.info("ℹ️ Long signal detected. Processing may take some time.")
    
    return valid

def create_signal_info_panel(signal: np.ndarray, fs: float) -> None:
    """
    Create informational panel with signal statistics.
    
    Args:
        signal: Audio signal
        fs: Sampling frequency (Hz)
    """
    if len(signal) == 0:
        return
    
    # Compute basic statistics
    duration = len(signal) / fs
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0
    
    # Display in expandable section
    with st.expander("📊 Signal Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{duration:.2f} s")
            st.metric("Samples", f"{len(signal):,}")
        
        with col2:
            st.metric("RMS Level", f"{rms:.4f}")
            st.metric("Peak Level", f"{peak:.4f}")
        
        with col3:
            st.metric("Crest Factor", f"{crest_factor:.2f}")
            st.metric("Sample Rate", f"{fs/1000:.1f} kHz")
        
        with col4:
            # Signal quality indicators
            if peak > 0.95:
                st.error("⚠️ Possible clipping")
            elif peak < 0.1:
                st.warning("⚠️ Low amplitude")
            else:
                st.success("✅ Good levels")
            
            if crest_factor > 10:
                st.warning("⚠️ High crest factor")
            else:
                st.success("✅ Normal dynamics")

def audio_format_converter(signal: np.ndarray, fs_in: float, 
                          fs_out: float) -> np.ndarray:
    """
    Convert audio between different sample rates.
    
    Args:
        signal: Input audio signal
        fs_in: Input sample rate (Hz)
        fs_out: Output sample rate (Hz)
        
    Returns:
        Resampled audio signal
    """
    if fs_in == fs_out:
        return signal
    
    try:
        import librosa
        signal_resampled = librosa.resample(signal, orig_sr=fs_in, target_sr=fs_out)
        return signal_resampled
    except ImportError:
        # Fallback to scipy
        from scipy import signal as sp_signal
        # Simple decimation/interpolation
        if fs_out < fs_in:
            # Decimation
            factor = int(fs_in // fs_out)
            return sp_signal.decimate(signal, factor)
        else:
            # Interpolation (zero-padding in frequency domain)
            factor = int(fs_out // fs_in)
            return np.repeat(signal, factor)

def create_audio_effects_panel(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Create panel with basic audio effects for testing.
    
    Args:
        signal: Input audio signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Modified signal
    """
    with st.expander("🎛️ Audio Effects (for testing)"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Gain control
            gain_db = st.slider("Gain (dB)", -20, 20, 0, 1)
            gain_linear = 10**(gain_db/20)
            
            # Add noise
            noise_level = st.slider("Noise Level", 0.0, 0.1, 0.0, 0.001)
        
        with col2:
            # Frequency shift
            freq_shift = st.slider("Frequency Shift (Hz)", -1000, 1000, 0, 10)
            
            # Apply effects checkbox
            apply_effects = st.checkbox("Apply Effects")
        
        if apply_effects:
            # Apply gain
            signal_modified = signal * gain_linear
            
            # Add noise
            if noise_level > 0:
                noise = np.random.randn(len(signal)) * noise_level
                signal_modified += noise
            
            # Frequency shift (simple method)
            if freq_shift != 0:
                t = np.arange(len(signal)) / fs
                shift_signal = np.cos(2*np.pi*freq_shift*t) + 1j*np.sin(2*np.pi*freq_shift*t)
                signal_complex = signal_modified * shift_signal
                signal_modified = np.real(signal_complex)
            
            # Ensure no clipping
            if np.max(np.abs(signal_modified)) > 1.0:
                signal_modified = signal_modified / np.max(np.abs(signal_modified)) * 0.95
                st.warning("⚠️ Signal normalized to prevent clipping")
            
            return signal_modified
        
        return signal