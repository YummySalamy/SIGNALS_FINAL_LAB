"""
Layout and UI Helper Functions

Provides reusable UI components and layout helpers for consistent
application appearance and functionality.

Features:
- Sidebar parameter controls
- Status indicators
- Progress bars
- Information panels
- Responsive layouts
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go

def create_sidebar_header(title: str, icon: str = "⚙️") -> None:
    """
    Create consistent sidebar header with icon.
    
    Args:
        title: Header title
        icon: Emoji icon
    """
    st.sidebar.markdown(f"## {icon} {title}")
    st.sidebar.markdown("---")

def create_parameter_section(title: str, icon: str = "🔧") -> None:
    """
    Create parameter section in sidebar.
    
    Args:
        title: Section title
        icon: Section icon
    """
    st.sidebar.markdown(f"### {icon} {title}")

def create_frequency_input(label: str = "Frequency (kHz)", 
                          min_val: float = 30.0,
                          max_val: float = 50.0, 
                          default: float = 40.0,
                          step: float = 0.1,
                          key: str = None) -> float:
    """
    Create standardized frequency input widget.
    
    Args:
        label: Input label
        min_val: Minimum value
        max_val: Maximum value
        default: Default value
        step: Step size
        key: Widget key
        
    Returns:
        Frequency value in Hz
    """
    freq_khz = st.sidebar.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step,
        key=key
    )
    return freq_khz * 1000  # Convert to Hz

def create_filter_controls(default_order: int = 101) -> Tuple[int, str, float]:
    """
    Create standardized filter control widgets.
    
    Args:
        default_order: Default filter order
        
    Returns:
        Tuple of (order, window_type, cutoff_ratio)
    """
    st.sidebar.markdown("### 🔧 Filter Parameters")
    
    # Limit maximum order to prevent overflow
    max_order = 501  # Reasonable maximum
    safe_default = min(default_order, max_order)
    
    order = st.sidebar.slider(
        "Filter Order:",
        min_value=51,
        max_value=max_order,
        value=safe_default,
        step=50,
        help="Higher order = sharper cutoff, more computation. >100 uses FIR automatically."
    )
    
    window_type = st.sidebar.selectbox(
        "Window Type:",
        ["hamming", "kaiser", "blackman"],
        index=0,
        help="Window function affects filter characteristics"
    )
    
    cutoff_ratio = st.sidebar.slider(
        "Cutoff Frequency (× fc):",
        min_value=0.1,
        max_value=2.0,
        value=0.3,
        step=0.1,
        help="Cutoff relative to carrier frequency"
    )
    
    # Display filter type info
    filter_type_info = "FIR" if order > 100 else "IIR"
    st.sidebar.info(f"ℹ️ Filter type: {filter_type_info}")
    
    return order, window_type, cutoff_ratio

def create_modulation_controls() -> Dict[str, Any]:
    """
    Create common modulation parameter controls.
    
    Returns:
        Dictionary of modulation parameters
    """
    st.sidebar.markdown("### 📡 Modulation Parameters")
    
    params = {}
    
    # Carrier frequency
    params['fc'] = create_frequency_input(
        "Carrier Frequency (kHz):",
        min_val=30.0,
        max_val=50.0,
        default=40.0
    )
    
    # Sampling frequency
    params['fs'] = st.sidebar.number_input(
        "Sampling Frequency (kHz):",
        min_value=80,
        max_value=1000,
        value=100,
        step=5,
        help="Must be > 2 × carrier frequency"
    ) * 1000
    
    # Validate Nyquist criterion
    if params['fc'] >= params['fs'] / 2:
        st.sidebar.error("⚠️ Carrier frequency exceeds Nyquist limit!")
    
    return params

def create_signal_selection_widget(demo_signals: List[str], 
                                  allow_upload: bool = True) -> Tuple[bool, str, Any]:
    """
    Create signal selection widget (demo vs upload).
    
    Args:
        demo_signals: List of available demo signals
        allow_upload: Whether to allow file upload
        
    Returns:
        Tuple of (use_demo, demo_type, uploaded_file)
    """
    st.sidebar.markdown("### 🎵 Signal Input")
    
    use_demo = st.sidebar.checkbox("Use demo signal", value=True)
    
    if use_demo:
        demo_type = st.sidebar.selectbox(
            "Demo signal type:",
            demo_signals,
            help="Select predefined signal for testing"
        )
        return True, demo_type, None
    
    elif allow_upload:
        uploaded_file = st.sidebar.file_uploader(
            "Upload audio file:",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, OGG"
        )
        return False, "", uploaded_file
    
    else:
        st.sidebar.warning("No upload option available")
        return True, demo_signals[0], None

def create_analysis_controls() -> Dict[str, Any]:
    """
    Create analysis and visualization controls.
    
    Returns:
        Dictionary of analysis parameters
    """
    st.sidebar.markdown("### 📊 Analysis Options")
    
    controls = {}
    
    controls['show_spectrum'] = st.sidebar.checkbox(
        "Show spectrum analysis", 
        value=True,
        help="Display frequency domain plots"
    )
    
    controls['show_metrics'] = st.sidebar.checkbox(
        "Show quality metrics", 
        value=True,
        help="Display SNR, THD, and other metrics"
    )
    
    controls['psd_method'] = st.sidebar.selectbox(
        "PSD Method:",
        ["welch", "periodogram", "multitaper"],
        index=0,
        help="Method for power spectral density estimation"
    )
    
    controls['time_window'] = st.sidebar.slider(
        "Time display window (ms):",
        min_value=10,
        max_value=200,
        value=50,
        help="Time window for waveform display"
    )
    
    return controls

def create_status_panel(processing: bool = False, 
                       message: str = "Ready",
                       progress: Optional[float] = None) -> None:
    """
    Create status panel with progress indication.
    
    Args:
        processing: Whether processing is active
        message: Status message
        progress: Progress value (0-1) if available
    """
    status_container = st.container()
    
    with status_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if processing:
                st.info(f"🔄 {message}")
            else:
                st.success(f"✅ {message}")
        
        with col2:
            if progress is not None:
                st.progress(progress)

def create_comparison_tabs(signals: Dict[str, np.ndarray],
                          fs: float,
                          time_labels: List[str] = None,
                          freq_labels: List[str] = None) -> None:
    """
    Create tabbed interface for signal comparison.
    
    Args:
        signals: Dictionary of signal_name -> signal_array
        fs: Sampling frequency
        time_labels: Custom labels for time domain tab
        freq_labels: Custom labels for frequency domain tab
    """
    if time_labels is None:
        time_labels = ["🕒 Time Domain", "📊 Frequency Domain", "🎵 Audio"]
    
    tabs = st.tabs(time_labels)
    
    # Time domain tab
    with tabs[0]:
        st.subheader("Waveforms")
        for name, signal in signals.items():
            if len(signal) > 0:
                # Show first 100ms or full signal if shorter
                n_display = min(int(0.1 * fs), len(signal))
                t = np.arange(n_display) / fs * 1000
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t, y=signal[:n_display],
                    mode='lines',
                    name=name,
                    hovertemplate=f'{name}<br>Time: %{{x:.2f}} ms<br>Amplitude: %{{y:.4f}}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{name} - Time Domain",
                    xaxis_title="Time (ms)",
                    yaxis_title="Amplitude",
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Frequency domain tab
    with tabs[1]:
        st.subheader("Power Spectral Density")
        # This would be implemented with actual PSD computation
        st.info("PSD plots would be generated here using the dsp.metrics module")
    
    # Audio tab (if applicable)
    if len(tabs) > 2:
        with tabs[2]:
            st.subheader("Audio Playback")
            for name, signal in signals.items():
                if len(signal) > 0:
                    st.markdown(f"**{name}:**")
                    st.audio(signal.astype(np.float32), sample_rate=int(fs))

def create_parameter_summary(params: Dict[str, Any]) -> None:
    """
    Create summary panel of current parameters.
    
    Args:
        params: Dictionary of parameters to display
    """
    with st.expander("📋 Current Parameters"):
        cols = st.columns(2)
        
        items = list(params.items())
        mid_point = len(items) // 2
        
        with cols[0]:
            for key, value in items[:mid_point]:
                if isinstance(value, float):
                    if 'freq' in key.lower() or 'fc' in key.lower():
                        st.text(f"{key}: {value/1000:.1f} kHz")
                    elif 'db' in key.lower():
                        st.text(f"{key}: {value:.1f} dB")
                    else:
                        st.text(f"{key}: {value:.3f}")
                else:
                    st.text(f"{key}: {value}")
        
        with cols[1]:
            for key, value in items[mid_point:]:
                if isinstance(value, float):
                    if 'freq' in key.lower() or 'fc' in key.lower():
                        st.text(f"{key}: {value/1000:.1f} kHz")
                    elif 'db' in key.lower():
                        st.text(f"{key}: {value:.1f} dB")
                    else:
                        st.text(f"{key}: {value:.3f}")
                else:
                    st.text(f"{key}: {value}")

def create_help_section(content: str, title: str = "❓ Help") -> None:
    """
    Create collapsible help section.
    
    Args:
        content: Help content (markdown)
        title: Section title
    """
    with st.expander(title):
        st.markdown(content)

def create_warning_panel(warnings: List[str]) -> None:
    """
    Create panel for displaying warnings and recommendations.
    
    Args:
        warnings: List of warning messages
    """
    if warnings:
        st.warning("⚠️ **Warnings:**")
        for warning in warnings:
            st.markdown(f"• {warning}")

def create_metrics_grid(metrics: Dict[str, float], 
                       units: Dict[str, str] = None,
                       thresholds: Dict[str, Tuple[str, float]] = None) -> None:
    """
    Create grid layout for displaying metrics with color coding.
    
    Args:
        metrics: Dictionary of metric_name -> value
        units: Dictionary of metric_name -> unit_string
        thresholds: Dictionary of metric_name -> (comparison, threshold_value)
                   comparison can be '>', '<', '>=', '<='
    """
    if not metrics:
        return
    
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    cols = st.columns(n_cols)
    
    for i, (name, value) in enumerate(metrics.items()):
        col_idx = i % n_cols
        
        with cols[col_idx]:
            # Format value
            if units and name in units:
                formatted_value = f"{value:.2f} {units[name]}"
            else:
                formatted_value = f"{value:.4f}"
            
            # Determine color based on thresholds
            if thresholds and name in thresholds:
                comparison, threshold = thresholds[name]
                
                if comparison == '>' and value > threshold:
                    delta_color = "normal"
                elif comparison == '<' and value < threshold:
                    delta_color = "normal" 
                elif comparison == '>=' and value >= threshold:
                    delta_color = "normal"
                elif comparison == '<=' and value <= threshold:
                    delta_color = "normal"
                else:
                    delta_color = "inverse"
            else:
                delta_color = "normal"
            
            st.metric(
                label=name,
                value=formatted_value,
                delta=None,
                delta_color=delta_color
            )

def create_processing_pipeline_diagram(stages: List[str]) -> None:
    """
    Create visual diagram of processing pipeline.
    
    Args:
        stages: List of processing stage names
    """
    st.markdown("### 🔄 Processing Pipeline")
    
    # Create horizontal flow diagram
    pipeline_text = " → ".join(stages)
    st.markdown(f"**{pipeline_text}**")
    
    # Create visual representation
    fig = go.Figure()
    
    x_positions = np.arange(len(stages))
    y_position = [0] * len(stages)
    
    # Add nodes
    for i, stage in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=[x_positions[i]], y=[0],
            mode='markers+text',
            marker=dict(size=40, color='lightblue'),
            text=stage,
            textposition="middle center",
            showlegend=False,
            hovertemplate=f'{stage}<extra></extra>'
        ))
    
    # Add arrows
    for i in range(len(stages) - 1):
        fig.add_annotation(
            x=x_positions[i] + 0.4, y=0,
            ax=x_positions[i] + 0.6, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black'
        )
    
    fig.update_layout(
        template='plotly_white',
        height=150,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_export_options(data: Dict[str, Any], 
                         formats: List[str] = ['CSV', 'JSON']) -> None:
    """
    Create export options for processed data.
    
    Args:
        data: Data dictionary to export
        formats: Available export formats
    """
    st.sidebar.markdown("### 💾 Export Options")
    
    export_format = st.sidebar.selectbox(
        "Export format:",
        formats,
        help="Choose format for exporting results"
    )
    
    if st.sidebar.button("📤 Export Data"):
        if export_format == 'CSV':
            # Convert to CSV format
            import pandas as pd
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="signal_analysis_results.csv",
                mime="text/csv"
            )
        
        elif export_format == 'JSON':
            # Convert to JSON format
            import json
            json_str = json.dumps(data, indent=2)
            st.sidebar.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name="signal_analysis_results.json",
                mime="application/json"
            )

def create_advanced_controls() -> Dict[str, Any]:
    """
    Create advanced/expert controls in collapsible section.
    
    Returns:
        Dictionary of advanced parameters
    """
    advanced = {}
    
    with st.sidebar.expander("🔬 Advanced Controls"):
        advanced['zero_phase'] = st.checkbox(
            "Zero-phase filtering",
            value=True,
            help="Use filtfilt for zero group delay"
        )
        
        advanced['window_overlap'] = st.slider(
            "Window overlap (%):",
            min_value=0,
            max_value=90,
            value=50,
            help="Overlap for spectral analysis windows"
        )
        
        advanced['fft_size'] = st.selectbox(
            "FFT Size:",
            [512, 1024, 2048, 4096, 8192],
            index=2,
            help="FFT size for frequency analysis"
        )
        
        advanced['precision'] = st.selectbox(
            "Numerical precision:",
            ["float32", "float64"],
            index=0,
            help="Numerical precision for calculations"
        )
    
    return advanced