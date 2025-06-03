"""
I/Q (Quadrature Amplitude Modulation) Page

Interactive demonstration of I/Q modulation exploiting orthogonality
of sine and cosine to transmit two signals simultaneously.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dsp.modulation import *
from dsp.filters import *
from dsp.metrics import *
from components.plotting import *
from components.audio_widgets import *
from components.layout import *
from utils.cache import *
from utils.file_io import *

st.set_page_config(page_title="I/Q QAM", page_icon="🔀", layout="wide")

def main():
    st.title("🔀 Modulación I/Q (Cuadratura de Amplitud)")
    
    st.markdown("""
    **Principio:** Aprovecha la ortogonalidad de seno y coseno para transmitir dos señales independientes 
    simultáneamente en la misma frecuencia portadora.
    
    **Ecuación:** s(t) = xI(t)cos(ωct) + xQ(t)sin(ωct)
    
    **Ortogonalidad:** ∫cos(ωct)sin(ωct)dt = 0
    """)
    
    # Sidebar controls
    create_sidebar_header("Configuración I/Q")
    
    # Signal inputs for I and Q channels
    create_parameter_section("Canal I (In-phase)")
    
    demo_signals_I = [
        "Tono 1 kHz", 
        "Suma de tonos graves", 
        "Chirp ascendente", 
        "Ruido rosa",
        "Voz masculina sintética"
    ]
    
    use_demo_I = st.sidebar.checkbox("Usar señal demo para I", value=True)
    
    if use_demo_I:
        demo_type_I = st.sidebar.selectbox(
            "Tipo señal I:",
            demo_signals_I,
            key="demo_I",
            help="Señal para canal en fase"
        )
    else:
        uploaded_file_I = st.sidebar.file_uploader(
            "Audio canal I:", 
            type=['wav', 'mp3', 'flac'], 
            key="upload_I",
            help="Archivo de audio para canal I"
        )
    
    # Q channel
    create_parameter_section("Canal Q (Quadrature)")
    
    demo_signals_Q = [
        "Tono 2 kHz", 
        "Suma de tonos agudos", 
        "Chirp descendente", 
        "Ruido blanco filtrado",
        "Voz femenina sintética"
    ]
    
    use_demo_Q = st.sidebar.checkbox("Usar señal demo para Q", value=True)
    
    if use_demo_Q:
        demo_type_Q = st.sidebar.selectbox(
            "Tipo señal Q:",
            demo_signals_Q,
            key="demo_Q",
            help="Señal para canal en cuadratura"
        )
    else:
        uploaded_file_Q = st.sidebar.file_uploader(
            "Audio canal Q:", 
            type=['wav', 'mp3', 'flac'], 
            key="upload_Q",
            help="Archivo de audio para canal Q"
        )
    
    # Common parameters
    duration = st.sidebar.slider("⏱️ Duración (s):", 1, 15, 5)
    
    # Modulation parameters
    mod_params = create_modulation_controls()
    fc = mod_params['fc']
    fs = mod_params['fs']
    
    # Filter parameters
    filter_order, filter_window, cutoff_ratio = create_filter_controls(301)
    
    # Phase error simulation
    st.sidebar.markdown("### 🔧 Simulación de errores")
    
    phase_error_deg = st.sidebar.slider(
        "Error de fase (grados):",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Simula desincronización del oscilador local"
    )
    
    freq_error_hz = st.sidebar.slider(
        "Error de frecuencia (Hz):",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        help="Simula deriva de frecuencia del oscilador"
    )
    
    # Analysis options
    analysis_opts = create_analysis_controls()
    
    # Advanced controls
    advanced = create_advanced_controls()
    
    # Generate I and Q signals
    with st.spinner("Generando señales I/Q..."):
        # I channel
        if use_demo_I:
            xI, fs_I = generate_iq_demo_signal(demo_type_I, duration, fs, channel='I')
        else:
            if uploaded_file_I is not None:
                xI, fs_I = load_and_preprocess_audio(uploaded_file_I, target_fs=fs)
            else:
                st.warning("⚠️ Sube archivo para canal I o activa señal demo")
                st.stop()
        
        # Q channel  
        if use_demo_Q:
            xQ, fs_Q = generate_iq_demo_signal(demo_type_Q, duration, fs, channel='Q')
        else:
            if uploaded_file_Q is not None:
                xQ, fs_Q = load_and_preprocess_audio(uploaded_file_Q, target_fs=fs)
            else:
                st.warning("⚠️ Sube archivo para canal Q o activa señal demo")
                st.stop()
    
    # Ensure same length and sampling rate
    min_len = min(len(xI), len(xQ))
    xI = xI[:min_len]
    xQ = xQ[:min_len]
    fs_actual = fs
    
    # Validate signals
    if not validate_audio_parameters(xI, fs_actual, fc) or not validate_audio_parameters(xQ, fs_actual, fc):
        st.stop()
    
    # Create processing pipeline diagram
    iq_stages = [
        "xI(t), xQ(t)", 
        "Moduladores I/Q", 
        "Suma s(t)", 
        "Canal", 
        "Demoduladores I/Q", 
        "Filtros PB", 
        "x̂I(t), x̂Q(t)"
    ]
    create_processing_pipeline_diagram(iq_stages)
    
    # Perform I/Q processing
    st.subheader("🔄 Procesamiento I/Q")
    
    with st.spinner("Procesando modulación y demodulación I/Q..."):
        # Create cache keys
        xI_hash = hash_array(xI)
        xQ_hash = hash_array(xQ)
        
        phase_error_rad = np.deg2rad(phase_error_deg)
        
        # Use cached I/Q processing
        try:
            results = cached_iq_processing(
                xI_hash, xQ_hash, fc, fs_actual, filter_order, phase_error_rad,
                _xI=xI, _xQ=xQ
            )
            s_transmitted = results['transmitted']
            xI_recovered = results['xI_recovered']
            xQ_recovered = results['xQ_recovered']
            sos = results['filter_sos']
        except:
            # Manual processing
            # Modulation
            s_transmitted = iq_modulate(xI, xQ, fc, fs_actual)
            
            # Add frequency error if specified
            if freq_error_hz != 0:
                t = np.arange(len(s_transmitted)) / fs_actual
                freq_drift = np.exp(1j * 2 * np.pi * freq_error_hz * t)
                s_transmitted_complex = s_transmitted + 1j * np.imag(signal.hilbert(s_transmitted))
                s_transmitted = np.real(s_transmitted_complex * freq_drift)
            
            # Design filter
            cutoff_freq = cutoff_ratio * fc
            sos = cached_filter_design(cutoff_freq, fs_actual, filter_order, 'iir', filter_window)
            
            # Demodulation
            xI_recovered, xQ_recovered = iq_demodulate(s_transmitted, fc, fs_actual, sos, phase_error_rad)
    
    # Compute quality metrics for both channels
    with st.spinner("Calculando métricas de calidad..."):
        # I channel metrics
        metrics_I = cached_metrics_computation(
            hash_array(xI), hash_array(xI_recovered), _signal1=xI, _signal2=xI_recovered
        )
        
        # Q channel metrics
        metrics_Q = cached_metrics_computation(
            hash_array(xQ), hash_array(xQ_recovered), _signal1=xQ, _signal2=xQ_recovered
        )
        
        # Cross-talk analysis
        crosstalk_I_to_Q = compute_crosstalk(xI, xQ_recovered)
        crosstalk_Q_to_I = compute_crosstalk(xQ, xI_recovered)
        
        # Phase error analysis
        if phase_error_deg != 0 or freq_error_hz != 0:
            phase_analysis = phase_error_analysis(xI, xQ, xI_recovered, xQ_recovered)
        else:
            phase_analysis = None
    
    # Display metrics dashboard
    st.subheader("📊 Métricas de Calidad I/Q")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Canal I (In-phase):**")
        metrics_I_dict = {
            "SNR I": f"{metrics_I['snr']:.1f} dB",
            "Correlación I": f"{metrics_I['correlation']:.4f}",
            "Error RMS I": f"{metrics_I['rms_error']:.6f}"
        }
        create_metrics_grid(metrics_I_dict)
    
    with col2:
        st.markdown("**📉 Canal Q (Quadrature):**")
        metrics_Q_dict = {
            "SNR Q": f"{metrics_Q['snr']:.1f} dB",
            "Correlación Q": f"{metrics_Q['correlation']:.4f}",
            "Error RMS Q": f"{metrics_Q['rms_error']:.6f}"
        }
        create_metrics_grid(metrics_Q_dict)
    
    # Cross-talk and isolation metrics
    st.markdown("**🔀 Análisis de aislamiento:**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔄 Crosstalk I→Q", f"{crosstalk_I_to_Q:.1f} dB")
    with col2:
        st.metric("🔄 Crosstalk Q→I", f"{crosstalk_Q_to_I:.1f} dB")
    with col3:
        isolation_avg = (crosstalk_I_to_Q + crosstalk_Q_to_I) / 2
        st.metric("🎯 Aislamiento promedio", f"{isolation_avg:.1f} dB")
    with col4:
        if phase_analysis:
            st.metric("📐 Error fase estimado", f"{phase_analysis['phase_error_deg']:.2f}°")
        else:
            st.metric("📐 Error fase", "0.00°")
    
    # Quality assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_snr = (metrics_I['snr'] + metrics_Q['snr']) / 2
        if avg_snr > 35:
            st.success("✅ Excelente calidad I/Q")
        elif avg_snr > 20:
            st.info("ℹ️ Buena calidad I/Q")
        else:
            st.warning("⚠️ Calidad I/Q regular")
    
    with col2:
        if isolation_avg < -20:
            st.success("✅ Excelente aislamiento")
        elif isolation_avg < -10:
            st.info("ℹ️ Aislamiento adecuado")
        else:
            st.warning("⚠️ Aislamiento insuficiente")
    
    with col3:
        if abs(phase_error_deg) < 1:
            st.success("✅ Sincronización perfecta")
        elif abs(phase_error_deg) < 5:
            st.info("ℹ️ Sincronización buena")
        else:
            st.warning("⚠️ Error de sincronización")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕒 Señales Temporales", 
        "📊 Análisis Espectral", 
        "🎯 Constelación I/Q",
        "🔬 Análisis de Errores",
        "🎵 Audio"
    ])
    
    with tab1:
        st.subheader("Señales en cada etapa del sistema I/Q")
        
        # Time vector for display
        display_duration = min(analysis_opts['time_window']/1000, len(xI)/fs_actual)
        n_display = int(display_duration * fs_actual)
        t_display = np.arange(n_display) / fs_actual * 1000
        
        # Create subplot for all signals
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Canal I Original", "Canal Q Original",
                "Señal Transmitida s(t)", "Señal Transmitida (zoom)",
                "Canal I Recuperado", "Canal Q Recuperado"
            ],
            vertical_spacing=0.1
        )
        
        # Original I and Q
        fig.add_trace(go.Scatter(
            x=t_display, y=xI[:n_display],
            mode='lines', name='xI(t)', line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=t_display, y=xQ[:n_display],
            mode='lines', name='xQ(t)', line=dict(color='red', width=2)
        ), row=1, col=2)
        
        # Transmitted signal
        fig.add_trace(go.Scatter(
            x=t_display, y=s_transmitted[:n_display],
            mode='lines', name='s(t)', line=dict(color='purple', width=1)
        ), row=2, col=1)
        
        # Transmitted signal zoom (first 5ms)
        n_zoom = min(int(0.005 * fs_actual), n_display)
        t_zoom = t_display[:n_zoom]
        fig.add_trace(go.Scatter(
            x=t_zoom, y=s_transmitted[:n_zoom],
            mode='lines', name='s(t) zoom', line=dict(color='purple', width=2)
        ), row=2, col=2)
        
        # Recovered I and Q
        fig.add_trace(go.Scatter(
            x=t_display, y=xI_recovered[:n_display],
            mode='lines', name='x̂I(t)', line=dict(color='darkblue', width=2, dash='dash')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=t_display, y=xQ_recovered[:n_display],
            mode='lines', name='x̂Q(t)', line=dict(color='darkred', width=2, dash='dash')
        ), row=3, col=2)
        
        # Add original signals as reference in recovered plots
        fig.add_trace(go.Scatter(
            x=t_display, y=xI[:n_display],
            mode='lines', name='xI original', line=dict(color='lightblue', width=1),
            opacity=0.5
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=t_display, y=xQ[:n_display],
            mode='lines', name='xQ original', line=dict(color='lightcoral', width=1),
            opacity=0.5
        ), row=3, col=2)
        
        fig.update_xaxes(title_text="Tiempo (ms)")
        fig.update_yaxes(title_text="Amplitud")
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error signals
        if st.checkbox("📊 Mostrar señales de error"):
            error_I = xI[:n_display] - xI_recovered[:n_display]
            error_Q = xQ[:n_display] - xQ_recovered[:n_display]
            
            fig_error = go.Figure()
            
            fig_error.add_trace(go.Scatter(
                x=t_display, y=error_I,
                mode='lines', name='Error I', line=dict(color='blue', width=2)
            ))
            
            fig_error.add_trace(go.Scatter(
                x=t_display, y=error_Q,
                mode='lines', name='Error Q', line=dict(color='red', width=2)
            ))
            
            fig_error.update_layout(
                title="Señales de Error I/Q",
                xaxis_title="Tiempo (ms)",
                yaxis_title="Error",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis espectral del sistema I/Q")
        
        # Compute PSDs
        f_I, psd_I = cached_psd_computation(
            hash_array(xI), fs_actual, analysis_opts['psd_method'], _signal=xI
        )
        f_Q, psd_Q = cached_psd_computation(
            hash_array(xQ), fs_actual, analysis_opts['psd_method'], _signal=xQ
        )
        f_tx, psd_tx = cached_psd_computation(
            hash_array(s_transmitted), fs_actual, analysis_opts['psd_method'], _signal=s_transmitted
        )
        f_I_rec, psd_I_rec = cached_psd_computation(
            hash_array(xI_recovered), fs_actual, analysis_opts['psd_method'], _signal=xI_recovered
        )
        f_Q_rec, psd_Q_rec = cached_psd_computation(
            hash_array(xQ_recovered), fs_actual, analysis_opts['psd_method'], _signal=xQ_recovered
        )
        
        # Create spectral comparison plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Espectros de banda base:**")
            
            fig_bb = go.Figure()
            
            # Original signals
            fig_bb.add_trace(go.Scatter(
                x=f_I/1000, y=psd_I,
                mode='lines', name='Canal I original', line=dict(color='blue', width=2)
            ))
            
            fig_bb.add_trace(go.Scatter(
                x=f_Q/1000, y=psd_Q,
                mode='lines', name='Canal Q original', line=dict(color='red', width=2)
            ))
            
            # Recovered signals
            fig_bb.add_trace(go.Scatter(
                x=f_I_rec/1000, y=psd_I_rec,
                mode='lines', name='Canal I recuperado', 
                line=dict(color='darkblue', width=2, dash='dash')
            ))
            
            fig_bb.add_trace(go.Scatter(
                x=f_Q_rec/1000, y=psd_Q_rec,
                mode='lines', name='Canal Q recuperado', 
                line=dict(color='darkred', width=2, dash='dash')
            ))
            
            fig_bb.update_layout(
                title="Espectros de Banda Base",
                xaxis_title="Frecuencia (kHz)",
                yaxis_title="Magnitud (dB)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_bb, use_container_width=True)
        
        with col2:
            st.markdown("**📡 Espectro de señal transmitida:**")
            
            fig_tx = create_psd_plot(f_tx, psd_tx, "Señal I/Q Transmitida", 'purple')
            
            # Add carrier frequency markers
            fig_tx.add_vline(x=fc/1000, line_dash="dash", line_color="red", 
                           annotation_text=f"fc = {fc/1000:.1f} kHz")
            
            st.plotly_chart(fig_tx, use_container_width=True)
        
        # Spectral efficiency analysis
        st.markdown("**🎯 Análisis de eficiencia espectral:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Estimate bandwidth usage
            bw_I = estimate_signal_bandwidth(f_I, psd_I)
            bw_Q = estimate_signal_bandwidth(f_Q, psd_Q)
            bw_combined = max(bw_I, bw_Q)
            
            st.metric("📏 Ancho banda I", f"{bw_I/1000:.2f} kHz")
            st.metric("📏 Ancho banda Q", f"{bw_Q/1000:.2f} kHz")
        
        with col2:
            # Spectral efficiency
            total_bw = 2 * bw_combined  # Double sideband
            efficiency = (bw_I + bw_Q) / total_bw * 100 if total_bw > 0 else 0
            
            st.metric("🎯 Eficiencia espectral", f"{efficiency:.1f}%")
            st.metric("📊 Ancho banda total", f"{total_bw/1000:.2f} kHz")
        
        with col3:
            # Power distribution
            power_I = np.mean(xI**2)
            power_Q = np.mean(xQ**2)
            power_ratio = power_I / power_Q if power_Q > 0 else np.inf
            
            st.metric("⚡ Potencia I/Q", f"{power_ratio:.2f}")
            if 0.5 < power_ratio < 2.0:
                st.success("✅ Balance de potencia")
            else:
                st.warning("⚠️ Desbalance de potencia")
    
    with tab3:
        st.subheader("Diagrama de constelación I/Q")
        
        # Create constellation plot
        fig_const = create_iq_constellation_plot(xI_recovered, xQ_recovered, 
                                                "Constelación I/Q Recuperada")
        
        st.plotly_chart(fig_const, use_container_width=True)
        
        # Constellation analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Estadísticas de constelación:**")
            
            # Compute constellation metrics
            const_metrics = constellation_analysis(xI_recovered + 1j*xQ_recovered, 'qam')
            
            st.text(f"Potencia promedio: {const_metrics['avg_power']:.4f}")
            st.text(f"PAPR: {const_metrics['papr_db']:.1f} dB")
            st.text(f"Correlación I/Q: {const_metrics['IQ_correlation']:.4f}")
            st.text(f"Desv. estándar I: {const_metrics['I_variance']**0.5:.4f}")
            st.text(f"Desv. estándar Q: {const_metrics['Q_variance']**0.5:.4f}")
        
        with col2:
            st.markdown("**🎯 Calidad de constelación:**")
            
            # EVM calculation
            reference_constellation = xI + 1j*xQ
            received_constellation = xI_recovered + 1j*xQ_recovered
            evm = compute_evm(reference_constellation, received_constellation)
            
            st.metric("📐 EVM", f"{evm:.2f}%")
            
            if evm < 5:
                st.success("✅ Excelente EVM")
            elif evm < 10:
                st.info("ℹ️ Buen EVM")
            else:
                st.warning("⚠️ EVM elevado")
            
            # Phase noise analysis
            if phase_analysis:
                st.text(f"Deriva de fase: {phase_analysis['phase_error_deg']:.2f}°")
                st.text(f"Aislamiento: {phase_analysis['isolation_dB']:.1f} dB")
    
    with tab4:
        st.subheader("Análisis detallado de errores")
        
        if phase_error_deg != 0 or freq_error_hz != 0:
            st.markdown("**🔬 Efectos de errores de sincronización:**")
            
            # Show effects of phase/frequency errors
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📐 Error de fase:**")
                if abs(phase_error_deg) > 0:
                    st.text(f"Error aplicado: {phase_error_deg:.1f}°")
                    st.text(f"Error estimado: {phase_analysis['phase_error_deg']:.2f}°")
                    
                    # Show theoretical crosstalk
                    theoretical_crosstalk = 20 * np.log10(abs(np.sin(np.deg2rad(phase_error_deg))))
                    st.text(f"Crosstalk teórico: {theoretical_crosstalk:.1f} dB")
                    st.text(f"Crosstalk medido: {isolation_avg:.1f} dB")
                else:
                    st.text("Sin error de fase")
            
            with col2:
                st.markdown("**📊 Error de frecuencia:**")
                if abs(freq_error_hz) > 0:
                    st.text(f"Error aplicado: {freq_error_hz:.1f} Hz")
                    st.text(f"Error relativo: {freq_error_hz/fc*1e6:.1f} ppm")
                    
                    # Estimate rotation rate
                    rotation_rate = 360 * freq_error_hz  # degrees per second
                    st.text(f"Rotación: {rotation_rate:.1f} °/s")
                else:
                    st.text("Sin error de frecuencia")
            
            # Error visualization
            if phase_analysis:
                st.markdown("**📈 Matriz de correlación cruzada:**")
                
                corr_matrix = np.array([
                    [phase_analysis['I_to_I_correlation'], phase_analysis['I_to_Q_leakage']],
                    [phase_analysis['Q_to_I_leakage'], phase_analysis['Q_to_Q_correlation']]
                ])
                
                fig_heatmap = create_error_heatmap(
                    20 * np.log10(np.abs(corr_matrix) + 1e-6),
                    ['I recuperado', 'Q recuperado'],
                    ['I original', 'Q original'],
                    "Matriz de Correlación (dB)"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        else:
            st.info("ℹ️ Sin errores aplicados. Ajusta los controles de error para ver sus efectos.")
        
        # Crosstalk analysis over time
        st.markdown("**🔀 Análisis temporal de crosstalk:**")
        
        if st.checkbox("📊 Mostrar crosstalk vs tiempo"):
            # Compute windowed crosstalk
            window_size = int(0.1 * fs_actual)  # 100ms windows
            n_windows = len(xI) // window_size
            
            crosstalk_time = []
            time_centers = []
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                window_I = xI[start_idx:end_idx]
                window_Q_rec = xQ_recovered[start_idx:end_idx]
                
                ct = compute_crosstalk(window_I, window_Q_rec)
                crosstalk_time.append(ct)
                time_centers.append((start_idx + end_idx) / 2 / fs_actual)
            
            fig_ct = go.Figure()
            fig_ct.add_trace(go.Scatter(
                x=time_centers, y=crosstalk_time,
                mode='lines+markers',
                name='Crosstalk I→Q',
                line=dict(color='red', width=2)
            ))
            
            fig_ct.update_layout(
                title="Evolución temporal del Crosstalk",
                xaxis_title="Tiempo (s)",
                yaxis_title="Crosstalk (dB)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_ct, use_container_width=True)
    
    with tab5:
        st.subheader("🎵 Reproducción y comparación de audio I/Q")
        
        # Audio players for all channels
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📻 Canal I:**")
            st.markdown("*Original:*")
            create_audio_player(xI, fs_actual, "audio_I_orig", show_waveform=False)
            
            st.markdown("*Recuperado:*")
            create_audio_player(xI_recovered, fs_actual, "audio_I_rec", show_waveform=False)
            
            # I channel download
            create_download_button(xI_recovered, fs_actual, 
                                 f"iq_channel_I_fc{fc/1000:.0f}kHz.wav", 
                                 "⬇️ Descargar Canal I")
        
        with col2:
            st.markdown("**📻 Canal Q:**")
            st.markdown("*Original:*")
            create_audio_player(xQ, fs_actual, "audio_Q_orig", show_waveform=False)
            
            st.markdown("*Recuperado:*")
            create_audio_player(xQ_recovered, fs_actual, "audio_Q_rec", show_waveform=False)
            
            # Q channel download
            create_download_button(xQ_recovered, fs_actual, 
                                 f"iq_channel_Q_fc{fc/1000:.0f}kHz.wav", 
                                 "⬇️ Descargar Canal Q")
        
        # Combined stereo output
        st.markdown("**🎧 Audio estéreo combinado (I=Left, Q=Right):**")
        
        # Create stereo signal
        min_len = min(len(xI_recovered), len(xQ_recovered))
        stereo_signal = np.column_stack([xI_recovered[:min_len], xQ_recovered[:min_len]])
        
        # Convert to interleaved format for audio player
        stereo_interleaved = stereo_signal.flatten()
        
        try:
            st.audio(audio_to_wav_bytes(stereo_interleaved, fs_actual), format='audio/wav')
        except:
            st.warning("⚠️ No se pudo reproducir audio estéreo")
        
        # Audio quality metrics
        st.markdown("**📊 Métricas de calidad de audio:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dynamic range
            dr_I = compute_papr(xI_recovered)
            dr_Q = compute_papr(xQ_recovered)
            
            st.metric("📈 Rango dinámico I", f"{dr_I:.1f} dB")
            st.metric("📈 Rango dinámico Q", f"{dr_Q:.1f} dB")
        
        with col2:
            # Perceptual quality estimate
            corr_I = metrics_I['correlation']
            corr_Q = metrics_Q['correlation']
            
            perceptual_quality_I = min(100, corr_I * 100)
            perceptual_quality_Q = min(100, corr_Q * 100)
            
            st.metric("🎵 Calidad perceptual I", f"{perceptual_quality_I:.1f}%")
            st.metric("🎵 Calidad perceptual Q", f"{perceptual_quality_Q:.1f}%")
    
    # Summary and recommendations
    st.markdown("---")
    st.subheader("📋 Resumen del Sistema I/Q")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Rendimiento alcanzado:**")
        avg_snr = (metrics_I['snr'] + metrics_Q['snr']) / 2
        avg_corr = (metrics_I['correlation'] + metrics_Q['correlation']) / 2
        
        st.markdown(f"• SNR promedio: **{avg_snr:.1f} dB**")
        st.markdown(f"• Correlación promedio: **{avg_corr:.3f}**")
        st.markdown(f"• Aislamiento I/Q: **{isolation_avg:.1f} dB**")
        st.markdown(f"• EVM: **{evm:.2f}%**")
        st.markdown(f"• Error de fase: **{phase_error_deg:.1f}°**")
    
    with col2:
        st.markdown("**💡 Recomendaciones:**")
        
        recommendations = []
        
        if avg_snr < 25:
            recommendations.append("• Aumentar orden del filtro")
        
        if isolation_avg > -15:
            recommendations.append("• Mejorar sincronización de fase")
        
        if evm > 8:
            recommendations.append("• Revisar balance de potencia I/Q")
        
        if abs(phase_error_deg) > 3:
            recommendations.append("• Calibrar oscilador local")
        
        if not recommendations:
            recommendations.append("• ✅ Sistema funcionando óptimamente")
        
        for rec in recommendations:
            st.markdown(rec)

# Helper functions for I/Q page

def generate_iq_demo_signal(signal_type: str, duration: float, fs: float, channel: str = 'I') -> Tuple[np.ndarray, float]:
    """Generate demo signals optimized for I/Q channels."""
    t = np.arange(0, duration, 1/fs)
    
    if channel == 'I':
        # I channel signals (typically lower frequencies)
        if signal_type == "Tono 1 kHz":
            signal = 0.8 * np.sin(2 * np.pi * 1000 * t)
        elif signal_type == "Suma de tonos graves":
            signal = (0.4 * np.sin(2 * np.pi * 300 * t) + 
                     0.3 * np.sin(2 * np.pi * 600 * t) + 
                     0.2 * np.sin(2 * np.pi * 900 * t))
        elif signal_type == "Chirp ascendente":
            signal = 0.7 * np.sin(2 * np.pi * (200 + 800 * t / duration) * t)
        elif signal_type == "Ruido rosa":
            white = np.random.randn(len(t))
            # Simple pink noise filter
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy import signal as sp_signal
            signal = 0.4 * sp_signal.lfilter(b, a, white)
        elif signal_type == "Voz masculina sintética":
            # Lower fundamental frequency for male voice
            signal = generate_synthetic_voice_iq(duration, fs, f0=100, formants=[800, 1200])
        else:
            signal = 0.8 * np.sin(2 * np.pi * 1000 * t)
    
    else:  # Q channel
        if signal_type == "Tono 2 kHz":
            signal = 0.8 * np.sin(2 * np.pi * 2000 * t)
        elif signal_type == "Suma de tonos agudos":
            signal = (0.4 * np.sin(2 * np.pi * 1500 * t) + 
                     0.3 * np.sin(2 * np.pi * 2000 * t) + 
                     0.2 * np.sin(2 * np.pi * 2500 * t))
        elif signal_type == "Chirp descendente":
            signal = 0.7 * np.sin(2 * np.pi * (3000 - 800 * t / duration) * t)
        elif signal_type == "Ruido blanco filtrado":
            white = np.random.randn(len(t))
            # High-pass filter for Q channel
            from scipy import signal as sp_signal
            sos = sp_signal.butter(4, 1000, btype='high', fs=fs, output='sos')
            signal = 0.3 * sp_signal.sosfilt(sos, white)
        elif signal_type == "Voz femenina sintética":
            # Higher fundamental frequency for female voice
            signal = generate_synthetic_voice_iq(duration, fs, f0=200, formants=[1000, 1800])
        else:
            signal = 0.8 * np.sin(2 * np.pi * 2000 * t)
    
    # Normalize and add some envelope variation
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.9
    
    # Add slight amplitude modulation for realism
    envelope = 1 + 0.1 * np.sin(2 * np.pi * 3 * t)
    signal *= envelope
    
    return signal.astype(np.float32), fs

def generate_synthetic_voice_iq(duration: float, fs: float, f0: float, formants: list) -> np.ndarray:
    """Generate synthetic voice for I/Q demonstration."""
    t = np.arange(0, duration, 1/fs)
    
    # Generate harmonic series with formant emphasis
    signal = np.zeros_like(t)
    
    for harmonic in range(1, 15):
        freq = harmonic * f0
        amplitude = 1 / harmonic**0.7  # More gradual rolloff than 1/n
        
        # Emphasize harmonics near formants
        for formant in formants:
            if abs(freq - formant) < formant * 0.4:
                amplitude *= 2.5
        
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some speech-like modulation
    speech_mod = 1 + 0.3 * np.sin(2 * np.pi * 7 * t)  # 7 Hz typical speech rate
    signal *= speech_mod
    
    # Add pauses for speech-like characteristics
    pause_mask = np.ones_like(t)
    pause_times = np.arange(1.0, duration, 2.0)  # Pause every 2 seconds
    for pause_time in pause_times:
        pause_start = int(pause_time * fs)
        pause_end = int((pause_time + 0.3) * fs)
        if pause_end < len(pause_mask):
            pause_mask[pause_start:pause_end] *= 0.1
    
    signal *= pause_mask
    
    return signal

if __name__ == "__main__":
    main()