"""
Interactive Plotting Components for Signal Analysis

Provides standardized plotting functions using Plotly for interactive
visualization of signals, spectra, and analysis results.

All plots follow consistent styling and include:
- Interactive zoom and pan
- Hover information
- Professional appearance
- Responsive design for Streamlit
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, List, Optional, Dict, Union

# Define consistent color palette
COLORS = {
    'primary': '#007acc',
    'secondary': '#ff6b35', 
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

def create_time_comparison_plot(t: np.ndarray, x_original: np.ndarray, 
                               x_reconstructed: np.ndarray, 
                               title: str = "Signal Comparison") -> go.Figure:
    """
    Create overlay plot comparing original and reconstructed signals.
    
    Args:
        t: Time vector (units: seconds, milliseconds, etc.)
        x_original: Original signal
        x_reconstructed: Reconstructed signal
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Original signal
    fig.add_trace(go.Scatter(
        x=t, y=x_original,
        mode='lines',
        name='Original',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='Original<br>t: %{x}<br>Amplitude: %{y:.4f}<extra></extra>'
    ))
    
    # Reconstructed signal
    fig.add_trace(go.Scatter(
        x=t, y=x_reconstructed,
        mode='lines',
        name='Reconstructed',
        line=dict(color=COLORS['secondary'], width=2, dash='dash'),
        hovertemplate='Reconstructed<br>t: %{x}<br>Amplitude: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title='Time',
        yaxis_title='Amplitude',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_line_spectrum_plot(coeffs: Dict, T: float) -> go.Figure:
    """
    Create stem plot for Fourier series line spectrum.
    
    Args:
        coeffs: Fourier coefficients dictionary
        T: Signal period
        
    Returns:
        Plotly figure with magnitude and phase spectra
    """
    N = coeffs['N']
    omega0 = coeffs['omega0']
    
    # Frequency vector (harmonics)
    freqs = np.arange(-N, N+1) * omega0 / (2*np.pi)  # Convert to Hz
    
    # Complex coefficients for two-sided spectrum
    c_coeffs = coeffs['c_coeffs']
    magnitudes = np.abs(c_coeffs)
    phases = np.angle(c_coeffs, deg=True)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Magnitude Spectrum |Cₖ|', 'Phase Spectrum ∠Cₖ'],
        vertical_spacing=0.12
    )
    
    # Magnitude spectrum (stem plot)
    for i, (f, mag) in enumerate(zip(freqs, magnitudes)):
        if mag > 1e-6:  # Only plot significant components
            fig.add_trace(go.Scatter(
                x=[f, f], y=[0, mag],
                mode='lines',
                line=dict(color=COLORS['primary'], width=3),
                showlegend=False,
                hovertemplate=f'f: {f:.3f} Hz<br>|Cₖ|: {mag:.4f}<extra></extra>'
            ), row=1, col=1)
            
            # Add markers at tips
            fig.add_trace(go.Scatter(
                x=[f], y=[mag],
                mode='markers',
                marker=dict(color=COLORS['primary'], size=8),
                showlegend=False,
                hovertemplate=f'f: {f:.3f} Hz<br>|Cₖ|: {mag:.4f}<extra></extra>'
            ), row=1, col=1)
    
    # Phase spectrum (only for significant components)
    for i, (f, mag, phase) in enumerate(zip(freqs, magnitudes, phases)):
        if mag > 1e-6:
            fig.add_trace(go.Scatter(
                x=[f, f], y=[0, phase],
                mode='lines',
                line=dict(color=COLORS['secondary'], width=3),
                showlegend=False,
                hovertemplate=f'f: {f:.3f} Hz<br>∠Cₖ: {phase:.1f}°<extra></extra>'
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=[f], y=[phase],
                mode='markers',
                marker=dict(color=COLORS['secondary'], size=8),
                showlegend=False,
                hovertemplate=f'f: {f:.3f} Hz<br>∠Cₖ: {phase:.1f}°<extra></extra>'
            ), row=2, col=1)
    
    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
    fig.update_yaxes(title_text='|Cₖ|', row=1, col=1)
    fig.update_yaxes(title_text='Phase (degrees)', row=2, col=1)
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

def create_error_convergence_plot(N_values: np.ndarray, 
                                 errors: np.ndarray) -> go.Figure:
    """
    Plot RMS error vs number of harmonics.
    
    Args:
        N_values: Array of harmonic numbers
        errors: Corresponding RMS errors
        
    Returns:
        Plotly figure showing convergence
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=N_values, y=errors,
        mode='lines+markers',
        name='RMS Error',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=6),
        hovertemplate='N: %{x}<br>RMS Error: %{y:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Convergence Analysis: RMS Error vs Number of Harmonics',
        xaxis_title='Number of Harmonics (N)',
        yaxis_title='RMS Error',
        yaxis_type='log',  # Log scale for better visualization
        template='plotly_white',
        height=400
    )
    
    return fig

def create_psd_plot(f: np.ndarray, psd_db: np.ndarray, 
                   title: str = "Power Spectral Density",
                   color: str = None) -> go.Figure:
    """
    Create power spectral density plot.
    
    Args:
        f: Frequency vector (Hz)
        psd_db: PSD in dB (normalized to peak)
        title: Plot title
        color: Line color (optional)
        
    Returns:
        Plotly figure
    """
    if color is None:
        color = COLORS['primary']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=f/1000,  # Convert to kHz for readability
        y=psd_db,
        mode='lines',
        name='PSD',
        line=dict(color=color, width=2),
        fill='tonexty' if np.min(psd_db) < -40 else None,
        hovertemplate='Frequency: %{x:.3f} kHz<br>Power: %{y:.1f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Frequency (kHz)',
        yaxis_title='Power (dB)',
        template='plotly_white',
        height=400,
        yaxis=dict(range=[max(-80, np.min(psd_db)-10), 5])  # Reasonable y-axis range
    )
    
    return fig

def create_modulation_chain_plot(signals: List[Tuple[np.ndarray, str]], 
                                t_display: np.ndarray) -> go.Figure:
    """
    Create subplot showing modulation/demodulation chain.
    
    Args:
        signals: List of (signal, label) tuples
        t_display: Time vector for display
        
    Returns:
        Plotly figure with subplots
    """
    n_signals = len(signals)
    
    fig = make_subplots(
        rows=n_signals, cols=1,
        subplot_titles=[label for _, label in signals],
        vertical_spacing=0.08
    )
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
              COLORS['warning'], COLORS['danger'], COLORS['info']]
    
    for i, (signal, label) in enumerate(signals):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=t_display * 1000,  # Convert to ms
            y=signal[:len(t_display)],
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'{label}<br>Time: %{{x:.2f}} ms<br>Amplitude: %{{y:.4f}}<extra></extra>'
        ), row=i+1, col=1)
    
    fig.update_xaxes(title_text='Time (ms)', row=n_signals, col=1)
    
    for i in range(n_signals):
        fig.update_yaxes(title_text='Amplitude', row=i+1, col=1)
    
    fig.update_layout(
        template='plotly_white',
        height=150 * n_signals,
        showlegend=False
    )
    
    return fig

def create_iq_constellation_plot(xI: np.ndarray, xQ: np.ndarray, 
                               title: str = "I/Q Constellation") -> go.Figure:
    """
    Create I/Q constellation diagram.
    
    Args:
        xI: In-phase component
        xQ: Quadrature component
        title: Plot title
        
    Returns:
        Plotly scatter plot
    """
    # Subsample for visualization (max 1000 points)
    if len(xI) > 1000:
        indices = np.linspace(0, len(xI)-1, 1000, dtype=int)
        xI_plot = xI[indices]
        xQ_plot = xQ[indices]
    else:
        xI_plot = xI
        xQ_plot = xQ
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=xI_plot, y=xQ_plot,
        mode='markers',
        marker=dict(
            size=4,
            color=np.arange(len(xI_plot)),
            colorscale='viridis',
            opacity=0.7
        ),
        name='Constellation',
        hovertemplate='I: %{x:.4f}<br>Q: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='In-phase (I)',
        yaxis_title='Quadrature (Q)',
        template='plotly_white',
        height=500,
        width=500,
        xaxis=dict(scaleanchor="y", scaleratio=1)  # Equal aspect ratio
    )
    
    return fig

def create_am_envelope_plot(t: np.ndarray, s_am: np.ndarray, 
                           envelope: np.ndarray, mu: float) -> go.Figure:
    """
    Plot AM signal with envelope detection.
    
    Args:
        t: Time vector
        s_am: AM modulated signal
        envelope: Detected envelope
        mu: Modulation index
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # AM signal
    fig.add_trace(go.Scatter(
        x=t*1000, y=s_am,
        mode='lines',
        name='AM Signal',
        line=dict(color=COLORS['primary'], width=1),
        opacity=0.7,
        hovertemplate='Time: %{x:.2f} ms<br>AM: %{y:.3f}<extra></extra>'
    ))
    
    # Envelope
    fig.add_trace(go.Scatter(
        x=t*1000, y=envelope,
        mode='lines',
        name='Envelope',
        line=dict(color=COLORS['danger'], width=3),
        hovertemplate='Time: %{x:.2f} ms<br>Envelope: %{y:.3f}<extra></extra>'
    ))
    
    # Negative envelope
    fig.add_trace(go.Scatter(
        x=t*1000, y=-envelope,
        mode='lines',
        name='Negative Envelope',
        line=dict(color=COLORS['danger'], width=3, dash='dash'),
        hovertemplate='Time: %{x:.2f} ms<br>-Envelope: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'AM Signal with Envelope Detection (μ = {mu:.1f})',
        xaxis_title='Time (ms)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    # Add annotation for overmodulation
    if mu > 1.0:
        fig.add_annotation(
            text="⚠️ Overmodulation Detected!",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(color=COLORS['danger'], size=14),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    return fig

def create_filter_response_plot(f: np.ndarray, H_mag: np.ndarray, 
                               H_phase: np.ndarray) -> go.Figure:
    """
    Plot filter frequency response (magnitude and phase).
    
    Args:
        f: Frequency vector (Hz)
        H_mag: Magnitude response (dB)
        H_phase: Phase response (degrees)
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Magnitude Response', 'Phase Response'],
        vertical_spacing=0.12
    )
    
    # Magnitude
    fig.add_trace(go.Scatter(
        x=f/1000, y=H_mag,
        mode='lines',
        name='Magnitude',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='Freq: %{x:.1f} kHz<br>Mag: %{y:.1f} dB<extra></extra>'
    ), row=1, col=1)
    
    # Phase
    fig.add_trace(go.Scatter(
        x=f/1000, y=H_phase,
        mode='lines',
        name='Phase',
        line=dict(color=COLORS['secondary'], width=2),
        hovertemplate='Freq: %{x:.1f} kHz<br>Phase: %{y:.1f}°<extra></extra>'
    ), row=2, col=1)
    
    fig.update_xaxes(title_text='Frequency (kHz)', row=2, col=1)
    fig.update_yaxes(title_text='Magnitude (dB)', row=1, col=1)
    fig.update_yaxes(title_text='Phase (degrees)', row=2, col=1)
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

def create_multiband_psd_plot(frequencies: List[np.ndarray], 
                             psds: List[np.ndarray],
                             labels: List[str],
                             title: str = "Spectral Comparison") -> go.Figure:
    """
    Create overlay plot of multiple PSDs for comparison.
    
    Args:
        frequencies: List of frequency vectors
        psds: List of PSD arrays (in dB)
        labels: List of signal labels
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
              COLORS['warning'], COLORS['danger'], COLORS['info']]
    
    for i, (f, psd, label) in enumerate(zip(frequencies, psds, labels)):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=f/1000,  # kHz
            y=psd,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'{label}<br>Freq: %{{x:.2f}} kHz<br>Power: %{{y:.1f}} dB<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Frequency (kHz)',
        yaxis_title='Power (dB)',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def create_error_heatmap(error_matrix: np.ndarray, 
                        x_labels: List[str], 
                        y_labels: List[str],
                        title: str = "Error Analysis") -> go.Figure:
    """
    Create heatmap for error analysis (e.g., crosstalk matrix).
    
    Args:
        error_matrix: 2D array of error values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis  
        title: Plot title
        
    Returns:
        Plotly heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=error_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlBu_r',
        hovertemplate='%{y} → %{x}<br>Error: %{z:.2f} dB<extra></extra>',
        colorbar=dict(title="Error (dB)")
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Received Channel',
        yaxis_title='Transmitted Channel',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_eye_diagram(signal: np.ndarray, symbol_period: int, 
                      oversample: int = 1) -> go.Figure:
    """
    Create eye diagram for digital signals.
    
    Args:
        signal: Digital signal
        symbol_period: Samples per symbol
        oversample: Oversampling factor
        
    Returns:
        Plotly figure with eye diagram
    """
    # Extract symbol periods
    n_symbols = len(signal) // symbol_period
    eye_data = []
    
    for i in range(min(n_symbols-1, 100)):  # Limit to 100 traces for performance
        start_idx = i * symbol_period
        end_idx = start_idx + 2 * symbol_period  # Two symbol periods
        if end_idx <= len(signal):
            eye_data.append(signal[start_idx:end_idx])
    
    fig = go.Figure()
    
    t_eye = np.arange(2 * symbol_period) / oversample
    
    for i, trace in enumerate(eye_data):
        fig.add_trace(go.Scatter(
            x=t_eye, y=trace,
            mode='lines',
            line=dict(color=COLORS['primary'], width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='Eye Diagram',
        xaxis_title='Time (samples)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_3d_waterfall_plot(time_segments: np.ndarray,
                            frequencies: np.ndarray,
                            spectrograms: np.ndarray) -> go.Figure:
    """
    Create 3D waterfall plot for time-frequency analysis.
    
    Args:
        time_segments: Time vector for segments
        frequencies: Frequency vector
        spectrograms: 2D array [time, frequency]
        
    Returns:
        Plotly 3D surface plot
    """
    fig = go.Figure(data=[go.Surface(
        x=frequencies/1000,  # kHz
        y=time_segments,
        z=spectrograms,
        colorscale='Viridis',
        hovertemplate='Time: %{y:.3f} s<br>Freq: %{x:.2f} kHz<br>Power: %{z:.1f} dB<extra></extra>'
    )])
    
    fig.update_layout(
        title='Spectrogram - Time-Frequency Analysis',
        scene=dict(
            xaxis_title='Frequency (kHz)',
            yaxis_title='Time (s)',
            zaxis_title='Power (dB)'
        ),
        template='plotly_white',
        height=600
    )
    
    return fig

def display_theoretical_background(example_id: str):
    """
    Display theoretical background for Fourier series examples.
    
    Args:
        example_id: Signal identifier ("3.6.1", "3.6.2", etc.)
    """
    if example_id == "3.6.1":
        st.markdown("""
        **📐 Onda Triangular (Ejemplo 3.6.1)**
        
        **Función:** x(t) = 1 - |t| para |t| ≤ 1, T = 2
        
        **Propiedades:**
        - Función par → solo coeficientes aₖ (bₖ = 0)
        - Continua pero no diferenciable en t = 0
        
        **Serie de Fourier:**
        ```
        x(t) = 1/2 + (8/π²)Σ[1/n²]cos(nπt)  (n impar)
        ```
        
        **Coeficientes analíticos:**
        - a₀ = 1
        - aₖ = 8/(π²k²) para k impar, 0 para k par
        - bₖ = 0 (función par)
        """)
        
    elif example_id == "3.6.2":
        st.markdown("""
        **📈 Onda Diente de Sierra (Ejemplo 3.6.2)**
        
        **Función:** x(t) = t para |t| ≤ π, T = 2π
        
        **Propiedades:**
        - Función impar → solo coeficientes bₖ (aₖ = a₀ = 0)
        - Discontinuidad en t = ±π → Fenómeno de Gibbs
        
        **Serie de Fourier:**
        ```
        x(t) = 2Σ[(-1)^(k+1)/k]sin(kt)
        ```
        
        **Fenómeno de Gibbs:**
        - Sobrepico del ~9% cerca de discontinuidades
        - No disminuye con más armónicos
        """)
        
    elif example_id == "3.6.3":
        st.markdown("""
        **📊 Onda Parabólica (Ejemplo 3.6.3)**
        
        **Función:** x(t) = t² para |t| ≤ π, T = 2π
        
        **Propiedades:**
        - Función par → solo coeficientes aₖ
        - Continua y diferenciable
        - Convergencia rápida
        
        **Serie de Fourier:**
        ```
        x(t) = π²/3 + 4Σ[(-1)^k/k²]cos(kt)
        ```
        
        **Convergencia:**
        - Más rápida que señales discontinuas
        - Error decrece como 1/N²
        """)
        
    elif example_id == "3.6.4":
        st.markdown("""
        **🔄 Señal Mixta (Ejemplo 3.6.4)**
        
        **Función:** 
        ```
        x(t) = t + 1  para -1 < t < 0
        x(t) = 1      para 0 < t < 1
        ```
        T = 2
        
        **Propiedades:**
        - Sin simetría → coeficientes aₖ y bₖ
        - Discontinuidad en t = 0
        - Salto finito → Fenómeno de Gibbs
        
        **Convergencia:**
        - Converge al promedio de límites laterales en discontinuidades
        - x(0⁺) = 1, x(0⁻) = 1 → Serie converge a 1
        """)

def create_metrics_dashboard(metrics: Dict) -> None:
    """
    Create a metrics dashboard with colored metric cards.
    
    Args:
        metrics: Dictionary of metric name-value pairs
    """
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < n_metrics:
                metric_name = list(metrics.keys())[idx]
                metric_value = list(metrics.values())[idx]
                
                with cols[col]:
                    # Format value based on type
                    if isinstance(metric_value, float):
                        if 'dB' in metric_name:
                            formatted_value = f"{metric_value:.1f} dB"
                        elif 'Hz' in metric_name:
                            formatted_value = f"{metric_value/1000:.1f} kHz"
                        elif '%' in metric_name:
                            formatted_value = f"{metric_value:.2f}%"
                        else:
                            formatted_value = f"{metric_value:.4f}"
                    else:
                        formatted_value = str(metric_value)
                    
                    st.metric(metric_name, formatted_value)