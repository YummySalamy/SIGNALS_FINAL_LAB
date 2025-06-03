"""
File I/O and Audio Processing Utilities

Handles audio file loading, saving, format conversion, and preprocessing
for signal processing applications.

Features:
- Multiple audio format support
- Automatic format conversion
- Sample rate conversion
- Audio preprocessing and validation
"""

import numpy as np
import soundfile as sf
import io
import tempfile
import warnings
from typing import Tuple, Optional, Union, List
from pathlib import Path
import streamlit as st

# Supported audio formats
SUPPORTED_FORMATS = {
    '.wav': 'WAV',
    '.mp3': 'MP3', 
    '.flac': 'FLAC',
    '.ogg': 'OGG',
    '.m4a': 'M4A',
    '.aac': 'AAC'
}

def load_and_preprocess_audio(uploaded_file, 
                             target_fs: Optional[float] = None,
                             max_duration: Optional[float] = None,
                             normalize: bool = True) -> Tuple[np.ndarray, float]:
    """
    Load and preprocess uploaded audio file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_fs: Target sampling frequency (Hz). If None, keep original
        max_duration: Maximum duration in seconds. If None, load entire file
        normalize: Whether to normalize amplitude to [-1, 1]
        
    Returns:
        Tuple of (processed_signal, sample_rate)
    """
    if uploaded_file is None:
        return np.array([]), 0.0
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio file
        try:
            signal, fs = sf.read(tmp_file_path)
        except Exception as e:
            # Try alternative loading methods for unsupported formats
            try:
                import librosa
                signal, fs = librosa.load(tmp_file_path, sr=target_fs, mono=True)
            except ImportError:
                raise ValueError(f"Cannot load {uploaded_file.name}. Install librosa for additional format support.")
        
        # Convert to mono if stereo
        if signal.ndim > 1:
            signal = convert_to_mono(signal)
            st.info("🔄 Converted stereo to mono")
        
        # Limit duration if specified
        if max_duration is not None and len(signal) / fs > max_duration:
            max_samples = int(max_duration * fs)
            signal = signal[:max_samples]
            st.info(f"⏱️ Truncated to {max_duration} seconds")
        
        # Resample if target frequency specified
        if target_fs is not None and fs != target_fs:
            signal = resample_signal(signal, fs, target_fs)
            fs = target_fs
            st.info(f"🔄 Resampled to {target_fs/1000:.1f} kHz")
        
        # Normalize amplitude
        if normalize and np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.95
        
        # Ensure float32 format
        signal = signal.astype(np.float32)
        
        # Validate result
        if len(signal) == 0:
            raise ValueError("Loaded signal is empty")
        
        return signal, fs
        
    except Exception as e:
        st.error(f"❌ Error loading audio file: {str(e)}")
        return np.array([]), 0.0

def convert_to_mono(signal: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Convert stereo/multichannel audio to mono.
    
    Args:
        signal: Input signal (channels x samples or samples x channels)
        method: Conversion method ('mean', 'left', 'right')
        
    Returns:
        Mono signal
    """
    if signal.ndim == 1:
        return signal  # Already mono
    
    # Determine channel arrangement (assume samples x channels if ambiguous)
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T  # Transpose to samples x channels
    
    if method == 'mean':
        return np.mean(signal, axis=1)
    elif method == 'left':
        return signal[:, 0]
    elif method == 'right':
        return signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
    else:
        return np.mean(signal, axis=1)

def resample_signal(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample signal to new sampling frequency.
    
    Args:
        signal: Input signal
        fs_in: Input sampling frequency (Hz)
        fs_out: Output sampling frequency (Hz)
        
    Returns:
        Resampled signal
    """
    if fs_in == fs_out:
        return signal
    
    try:
        # Try librosa first (highest quality)
        import librosa
        return librosa.resample(signal, orig_sr=fs_in, target_sr=fs_out)
        
    except ImportError:
        # Fallback to scipy
        from scipy import signal as sp_signal
        
        # Calculate resampling ratio
        ratio = fs_out / fs_in
        
        if ratio > 1:
            # Upsampling: interpolate
            up_factor = int(np.ceil(ratio))
            down_factor = int(np.ceil(up_factor / ratio))
            return sp_signal.resample_poly(signal, up_factor, down_factor)
        else:
            # Downsampling: decimate
            down_factor = int(np.ceil(1 / ratio))
            up_factor = int(np.ceil(down_factor * ratio))
            return sp_signal.resample_poly(signal, up_factor, down_factor)

def save_audio_file(signal: np.ndarray, fs: float, filename: str,
                   format: str = 'WAV', subtype: str = 'PCM_16') -> bool:
    """
    Save audio signal to file.
    
    Args:
        signal: Audio signal
        fs: Sampling frequency (Hz)
        filename: Output filename
        format: Audio format ('WAV', 'FLAC', etc.)
        subtype: Audio subtype ('PCM_16', 'FLOAT', etc.)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure signal is in valid range
        if np.max(np.abs(signal)) > 1.0:
            signal = signal / np.max(np.abs(signal)) * 0.95
            warnings.warn("Signal clipped to prevent overflow")
        
        # Convert to appropriate format
        if subtype == 'PCM_16':
            signal_out = (signal * 32767).astype(np.int16)
        elif subtype == 'PCM_24':
            signal_out = (signal * 8388607).astype(np.int32)
        else:
            signal_out = signal.astype(np.float32)
        
        # Save file
        sf.write(filename, signal_out, int(fs), format=format, subtype=subtype)
        return True
        
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return False

def audio_file_info(file_path: str) -> dict:
    """
    Get information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with file information
    """
    try:
        info = sf.info(file_path)
        
        return {
            'duration': info.frames / info.samplerate,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
            'format': info.format,
            'subtype': info.subtype,
            'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        return {'error': str(e)}

def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file format and content.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check file exists
        if not Path(file_path).exists():
            return False, "File does not exist"
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}"
        
        # Try to get file info
        info = sf.info(file_path)
        
        # Check basic properties
        if info.frames == 0:
            return False, "File contains no audio data"
        
        if info.samplerate < 8000:
            return False, "Sample rate too low (< 8 kHz)"
        
        if info.samplerate > 192000:
            return False, "Sample rate too high (> 192 kHz)"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_test_signals(duration: float = 5.0, fs: float = 48000) -> dict:
    """
    Create test signals for development and testing.
    
    Args:
        duration: Signal duration in seconds
        fs: Sampling frequency (Hz)
        
    Returns:
        Dictionary of test signals
    """
    t = np.arange(0, duration, 1/fs)
    signals = {}
    
    # Pure tone
    signals['tone_1khz'] = 0.8 * np.sin(2 * np.pi * 1000 * t)
    
    # Multi-tone
    signals['multi_tone'] = (0.4 * np.sin(2 * np.pi * 440 * t) +
                           0.3 * np.sin(2 * np.pi * 880 * t) +
                           0.2 * np.sin(2 * np.pi * 1760 * t))
    
    # Chirp (frequency sweep)
    f0, f1 = 100, 4000
    signals['chirp'] = 0.7 * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    
    # White noise
    signals['white_noise'] = 0.3 * np.random.randn(len(t))
    
    # Pink noise (1/f spectrum)
    white = np.random.randn(len(t))
    # Simple pink noise approximation
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy import signal
    signals['pink_noise'] = 0.3 * signal.lfilter(b, a, white)
    
    # AM modulated signal
    carrier_freq = 5000
    mod_freq = 100
    mod_depth = 0.5
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulator = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    signals['am_signal'] = 0.6 * modulator * carrier
    
    return signals

def export_processing_results(results: dict, filename: str = "processing_results") -> None:
    """
    Export processing results to various formats.
    
    Args:
        results: Dictionary containing processing results
        filename: Base filename (without extension)
    """
    # Export as numpy arrays
    try:
        np.savez(f"{filename}.npz", **results)
        st.success(f"✅ Results saved as {filename}.npz")
    except Exception as e:
        st.error(f"Error saving NPZ file: {str(e)}")
    
    # Export audio signals as WAV files
    for key, value in results.items():
        if isinstance(value, np.ndarray) and 'signal' in key.lower():
            try:
                # Assume 48 kHz if not specified
                fs = results.get('sample_rate', 48000)
                save_audio_file(value, fs, f"{filename}_{key}.wav")
            except Exception as e:
                st.warning(f"Could not save {key} as audio: {str(e)}")

def batch_audio_processing(file_list: List[str], 
                          process_func: callable,
                          output_dir: str = "processed") -> List[str]:
    """
    Process multiple audio files in batch.
    
    Args:
        file_list: List of input file paths
        process_func: Processing function that takes (signal, fs) and returns processed signal
        output_dir: Output directory for processed files
        
    Returns:
        List of output file paths
    """
    output_files = []
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(file_list):
        try:
            status_text.text(f"Processing {Path(file_path).name}...")
            
            # Load audio
            signal, fs = sf.read(file_path)
            
            # Convert to mono if needed
            if signal.ndim > 1:
                signal = convert_to_mono(signal)
            
            # Process signal
            processed_signal = process_func(signal, fs)
            
            # Save processed audio
            output_path = Path(output_dir) / f"processed_{Path(file_path).name}"
            save_audio_file(processed_signal, fs, str(output_path))
            output_files.append(str(output_path))
            
            # Update progress
            progress_bar.progress((i + 1) / len(file_list))
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
    
    status_text.text("Batch processing complete!")
    return output_files

def get_audio_format_info() -> dict:
    """
    Get information about supported audio formats.
    
    Returns:
        Dictionary with format information
    """
    return {
        'formats': SUPPORTED_FORMATS,
        'recommended_format': 'WAV',
        'recommended_sample_rates': [44100, 48000, 96000],
        'recommended_bit_depths': [16, 24, 32],
        'max_file_size_mb': 100,
        'max_duration_minutes': 10
    }

def create_audio_file_uploader_advanced(key: str = None) -> Optional[Tuple[np.ndarray, float, dict]]:
    """
    Advanced audio file uploader with detailed information display.
    
    Args:
        key: Streamlit widget key
        
    Returns:
        Tuple of (signal, sample_rate, file_info) or None
    """
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=list(SUPPORTED_FORMATS.keys()),
        key=key,
        help="Drag and drop or browse for audio files"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_info = {
            'name': uploaded_file.name,
            'size_mb': uploaded_file.size / (1024 * 1024),
            'type': uploaded_file.type
        }
        
        # Load and process audio
        signal, fs = load_and_preprocess_audio(uploaded_file)
        
        if len(signal) > 0:
            # Update file info with audio properties
            file_info.update({
                'duration': len(signal) / fs,
                'sample_rate': fs,
                'samples': len(signal),
                'peak_amplitude': np.max(np.abs(signal)),
                'rms_level': np.sqrt(np.mean(signal**2))
            })
            
            # Display file information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{file_info['duration']:.1f} s")
                st.metric("File Size", f"{file_info['size_mb']:.1f} MB")
            with col2:
                st.metric("Sample Rate", f"{file_info['sample_rate']/1000:.1f} kHz")
                st.metric("Samples", f"{file_info['samples']:,}")
            with col3:
                st.metric("Peak Level", f"{file_info['peak_amplitude']:.3f}")
                st.metric("RMS Level", f"{file_info['rms_level']:.3f}")
            
            return signal, fs, file_info
    
    return None