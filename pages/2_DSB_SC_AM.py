"""
DSB-SC (Double Sideband Suppressed Carrier) Amplitude Modulation Page

Interactive demonstration of DSB-SC modulation and coherent demodulation
with real-time parameter adjustment and signal analysis.
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

st.set_page_config(page_title="DSB-SC AM", page_icon="📡", layout="wide")

def main():
    st.title("📡 Modulación DSB-SC (Doble Banda Lateral)")
    
    st.markdown("""
    **Principio:** La modulación DSB-SC multiplica la señal mensaje por una portadora coseno.
    El espectro se desplaza a ±fc sin desperdiciar potencia en la portadora.
    
    **Ecuación:** y(t) = x(t) × cos(ωct) ↔ Y(f) = ½[X(f-fc) + X(f+fc)]
    """)
    
    # Sidebar controls
    create_sidebar_header("Configuración del Sistema")
    
    # Signal input section
    create_parameter_section("Señal de Entrada")
    
    demo_signals = [
        "Tono puro 1kHz", 
        "Suma de tonos", 
        "Chirp (barrido)", 
        "Ruido filtrado",
        "Voz sintética"
    ]
    
    use_demo, demo_type, uploaded_file = create_signal_selection_widget(demo_signals)
    
    # Signal generation parameters
    if use_demo:
        duration = st.sidebar.slider("⏱️ Duración (s):", 1, 15, 5)
        
        if demo_type == "Voz sintética":
            # Parameters for synthetic voice
            voice_freq = st.sidebar.slider("🗣️ Frecuencia fundamental (Hz):", 80, 300, 120)
            voice_formants = st.sidebar.multiselect(
                "🎵 Formantes (Hz):", 
                [800, 1200, 2400, 3400], 
                default=[800, 1200]
            )
    
    # Modulation parameters
    mod_params = create_modulation_controls()
    fc = mod_params['fc']
    fs = mod_params['fs']
    
    # Filter parameters
    filter_order, filter_window, cutoff_ratio = create_filter_controls(501)
    
    # Analysis options
    analysis_opts = create_analysis_controls()
    
    # Advanced controls
    advanced = create_advanced_controls()
    
    # Validate Nyquist criterion
    if fc >= fs/2:
        st.error("❌ La frecuencia portadora excede el límite de Nyquist!")
        st.stop()
    
    # Generate or load signal
    with st.spinner("Generando/cargando señal..."):
        if use_demo:
            if demo_type == "Voz sintética":
                # Generate synthetic voice signal
                x, fs_actual = generate_synthetic_voice(duration, fs, voice_freq, voice_formants)
            else:
                # Use standard demo signals
                x, fs_actual = cached_demo_signal(demo_type, duration, fs)
        else:
            if uploaded_file is not None:
                # Use the simpler load function instead of the advanced one
                x, fs_actual = load_and_preprocess_audio(uploaded_file)
                if len(x) == 0:
                    st.warning("⚠️ Error cargando el archivo de audio")
                    st.stop()
            else:
                st.warning("⚠️ Selecciona una señal demo o sube un archivo")
                st.stop()
    
    # Resample if needed for RF processing
    if fs_actual < 2.2 * fc:
        fs_rf = int(2.5 * fc)
        x = resample_signal(x, fs_actual, fs_rf)
        fs_actual = fs_rf
        st.info(f"🔄 Señal remuestreada a {fs_rf/1000:.1f} kHz para evitar aliasing")
    
    # Validate audio parameters
    if not validate_audio_parameters(x, fs_actual, fc):
        st.stop()
    
    # Create processing pipeline diagram
    processing_stages = [
        "Señal x(t)", 
        "Modulador ×cos(ωct)", 
        "Canal y(t)", 
        "Mezclador ×2cos(ωct)", 
        "Filtro PB", 
        "Señal x̂(t)"
    ]
    create_processing_pipeline_diagram(processing_stages)
    
    # Perform DSB-SC processing
    st.subheader("🔄 Procesamiento DSB-SC")
    
    with st.spinner("Procesando modulación y demodulación..."):
        # Create cache keys
        signal_hash = hash_array(x)
        cache_key = f"{signal_hash}_{fc}_{fs_actual}_{filter_order}_{cutoff_ratio}"
        
        # Use cached processing if available
        try:
            results = cached_modulation_demodulation(
                signal_hash, fc, fs_actual, filter_order, 'dsb_sc', _signal=x
            )
            y_modulated = results['modulated']
            x_recovered = results['demodulated']
            sos = results['filter_sos']
        except:
            # Manual processing if cache fails
            # Modulation
            y_modulated = dsb_sc_modulate(x, fc, fs_actual)
            
            # Design receiver filter
            cutoff_freq = cutoff_ratio * fc
            # Use FIR for high orders to avoid overflow
            filter_type = 'fir' if filter_order > 20 else 'iir'
            sos = cached_filter_design(cutoff_freq, fs_actual, filter_order, filter_type, filter_window)
            
            # Demodulation steps
            x_mixed = dsb_sc_demod_mix(y_modulated, fc, fs_actual)
            
            # Apply filter
            if advanced['zero_phase']:
                x_recovered = apply_iir_filter(x_mixed, sos, zero_phase=True)
            else:
                x_recovered = apply_iir_filter(x_mixed, sos, zero_phase=False)
    
    # Compute quality metrics
    with st.spinner("Calculando métricas de calidad..."):
        metrics = cached_metrics_computation(
            hash_array(x), hash_array(x_recovered), _signal1=x, _signal2=x_recovered
        )
        
        snr = metrics['snr']
        correlation = metrics['correlation']
        rms_error = metrics['rms_error']
    
    # Display metrics dashboard
    st.subheader("📊 Métricas de Calidad")
    
    metrics_dict = {
        "SNR": snr,  # Pass raw values, not formatted strings
        "Correlación": correlation,
        "Error RMS": rms_error,
        "fc (kHz)": fc/1000,
        "fs (kHz)": fs_actual/1000,
        "Orden filtro": filter_order
    }
    
    # Define units for proper formatting
    units_dict = {
        "SNR": "dB",
        "fc (kHz)": "kHz", 
        "fs (kHz)": "kHz"
    }
    
    create_metrics_grid(metrics_dict, units_dict)
    
    # Quality assessment
    col1, col2, col3 = st.columns(3)
    with col1:
        if snr > 40:
            st.success("✅ Excelente calidad")
        elif snr > 20:
            st.info("ℹ️ Buena calidad")
        else:
            st.warning("⚠️ Calidad regular")
    
    with col2:
        if correlation > 0.95:
            st.success("✅ Alta correlación")
        elif correlation > 0.8:
            st.info("ℹ️ Correlación moderada")
        else:
            st.warning("⚠️ Baja correlación")
    
    with col3:
        efficiency = min(snr/60*100, 100)  # Normalize to 60 dB max
        st.metric("🎯 Eficiencia", f"{efficiency:.1f}%")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕒 Dominio Temporal", 
        "📊 Dominio Frecuencial", 
        "🎛️ Respuesta del Filtro",
        "🎵 Audio"
    ])
    
    with tab1:
        st.subheader("Formas de onda en cada etapa del sistema")
        
        # Advanced visualization controls
        st.markdown("**🎛️ Controles de visualización:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            time_unit = st.selectbox(
                "⏱️ Unidad de tiempo:",
                ["Milisegundos", "Segundos"],
                index=0,
                key="time_unit"
            )
        
        with col2:
            if time_unit == "Milisegundos":
                max_duration_ms = len(x) / fs_actual * 1000
                display_duration_ms = st.slider(
                    "📏 Duración a mostrar (ms):",
                    min_value=10,
                    max_value=min(int(max_duration_ms), 2000),
                    value=min(int(analysis_opts['time_window']), int(max_duration_ms)),
                    step=10
                )
                display_duration = display_duration_ms / 1000
            else:
                max_duration_s = len(x) / fs_actual
                display_duration = st.slider(
                    "📏 Duración a mostrar (s):",
                    min_value=0.1,
                    max_value=min(max_duration_s, 10.0),
                    value=min(2.0, max_duration_s),
                    step=0.1
                )
        
        with col3:
            if time_unit == "Milisegundos":
                zoom_start_ms = st.slider(
                    "🔍 Inicio (ms):",
                    min_value=0,
                    max_value=max(0, int(max_duration_ms - display_duration_ms)),
                    value=0,
                    step=10
                )
                zoom_start = zoom_start_ms / 1000
            else:
                zoom_start = st.slider(
                    "🔍 Inicio (s):",
                    min_value=0.0,
                    max_value=max(0.0, max_duration_s - display_duration),
                    value=0.0,
                    step=0.1
                )
        
        with col4:
            show_envelope = st.checkbox("📈 Mostrar envolvente", value=False)
        
        # Calculate display parameters
        start_sample = int(zoom_start * fs_actual)
        n_display = int(display_duration * fs_actual)
        end_sample = min(start_sample + n_display, len(x))
        
        if time_unit == "Milisegundos":
            t_display = np.arange(start_sample, end_sample) / fs_actual * 1000
            time_label = "Tiempo (ms)"
        else:
            t_display = np.arange(start_sample, end_sample) / fs_actual
            time_label = "Tiempo (s)"
        
        # Create modulation chain signals
        signals = [
            (x[start_sample:end_sample], "Señal original x(t)"),
            (y_modulated[start_sample:end_sample], "Señal modulada y(t) = x(t)cos(ωct)"),
            (x_recovered[start_sample:end_sample], "Señal recuperada x̂(t)")
        ]
        
        # If we have intermediate signals, add them
        if 'x_mixed' in locals():
            signals.insert(2, (x_mixed[start_sample:end_sample], "Después del mezclador: 2y(t)cos(ωct)"))
        
        fig_chain = create_modulation_chain_plot(signals, t_display)
        fig_chain.update_xaxes(title_text=time_label)
        
        # Add envelope if requested
        if show_envelope and len(signals) >= 2:
            # Calculate envelope of modulated signal
            modulated_signal = signals[1][0]
            envelope = np.abs(signal.hilbert(modulated_signal))
            
            # Add envelope traces to the modulated signal subplot
            fig_chain.add_trace(go.Scatter(
                x=t_display[:len(envelope)], 
                y=envelope,
                mode='lines',
                name='Envolvente +',
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2'
            ), row=2, col=1)
            
            fig_chain.add_trace(go.Scatter(
                x=t_display[:len(envelope)], 
                y=-envelope,
                mode='lines',
                name='Envolvente -',
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2'
            ), row=2, col=1)
        
        st.plotly_chart(fig_chain, use_container_width=True)
        
        # Information panel
        st.markdown("**📊 Información de la vista actual:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Muestras mostradas", f"{end_sample - start_sample:,}")
        with col2:
            if time_unit == "Milisegundos":
                st.metric("⏱️ Duración", f"{display_duration*1000:.1f} ms")
            else:
                st.metric("⏱️ Duración", f"{display_duration:.2f} s")
        with col3:
            st.metric("📍 Posición", f"{zoom_start/len(x)*fs_actual*100:.1f}%")
        with col4:
            resolution = (end_sample - start_sample) / (t_display[-1] - t_display[0]) if len(t_display) > 1 else 0
            if time_unit == "Milisegundos":
                st.metric("🎯 Resolución", f"{resolution:.1f} pts/ms")
            else:
                st.metric("🎯 Resolución", f"{resolution/1000:.1f} pts/ms")
        
        # Quick navigation buttons
        st.markdown("**⚡ Navegación rápida:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⏮️ Inicio"):
                st.rerun()
        
        with col2:
            if st.button("⬅️ Anterior") and zoom_start > 0:
                new_start = max(0, zoom_start - display_duration)
                st.session_state.zoom_start = new_start
                st.rerun()
        
        with col3:
            if st.button("🎯 Centro"):
                center_time = len(x) / fs_actual / 2
                new_start = max(0, center_time - display_duration/2)
                st.session_state.zoom_start = new_start
                st.rerun()
        
        with col4:
            if st.button("➡️ Siguiente") and zoom_start + display_duration < len(x)/fs_actual:
                new_start = min(len(x)/fs_actual - display_duration, zoom_start + display_duration)
                st.session_state.zoom_start = new_start
                st.rerun()
                
        with col5:
            if st.button("⏭️ Final"):
                new_start = max(0, len(x)/fs_actual - display_duration)
                st.session_state.zoom_start = new_start
                st.rerun()
        
        # Detailed comparison view
        if st.checkbox("🔬 Vista de comparación detallada"):
            st.markdown("**📊 Comparación Original vs Recuperada:**")
            
            fig_comparison = go.Figure()
            
            # Original signal
            fig_comparison.add_trace(go.Scatter(
                x=t_display, y=x[start_sample:end_sample],
                mode='lines+markers' if len(t_display) < 200 else 'lines',
                name='Original x(t)',
                line=dict(color='blue', width=2),
                marker=dict(size=3) if len(t_display) < 200 else None
            ))
            
            # Recovered signal
            fig_comparison.add_trace(go.Scatter(
                x=t_display, y=x_recovered[start_sample:end_sample],
                mode='lines+markers' if len(t_display) < 200 else 'lines',
                name='Recuperada x̂(t)',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=3) if len(t_display) < 200 else None
            ))
            
            # Error signal
            error_signal = x[start_sample:end_sample] - x_recovered[start_sample:end_sample]
            fig_comparison.add_trace(go.Scatter(
                x=t_display, y=error_signal,
                mode='lines',
                name='Error',
                line=dict(color='green', width=1),
                yaxis='y2'
            ))
            
            fig_comparison.update_layout(
                title="Comparación Detallada con Error",
                xaxis_title=time_label,
                yaxis_title="Amplitud Señal",
                yaxis2=dict(
                    title="Error",
                    overlaying="y",
                    side="right",
                    range=[np.min(error_signal)*1.1, np.max(error_signal)*1.1]
                ),
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Error statistics for current view
            col1, col2, col3 = st.columns(3)
            with col1:
                local_snr = compute_snr(x[start_sample:end_sample], x_recovered[start_sample:end_sample])
                st.metric("📊 SNR local", f"{local_snr:.1f} dB")
            with col2:
                local_rms = np.sqrt(np.mean(error_signal**2))
                st.metric("❌ RMS local", f"{local_rms:.6f}")
            with col3:
                local_corr = np.corrcoef(x[start_sample:end_sample], x_recovered[start_sample:end_sample])[0,1]
                st.metric("🔗 Correlación local", f"{local_corr:.4f}")
    
    with tab2:
        st.subheader("Análisis espectral del sistema DSB-SC")
        
        if analysis_opts['show_spectrum']:
            # Compute PSDs for all signals
            f_orig, psd_orig = cached_psd_computation(
                hash_array(x), fs_actual, analysis_opts['psd_method'], _signal=x
            )
            f_mod, psd_mod = cached_psd_computation(
                hash_array(y_modulated), fs_actual, analysis_opts['psd_method'], _signal=y_modulated
            )
            f_rec, psd_rec = cached_psd_computation(
                hash_array(x_recovered), fs_actual, analysis_opts['psd_method'], _signal=x_recovered
            )
            
            # Create multi-band PSD plot
            frequencies = [f_orig, f_mod, f_rec]
            psds = [psd_orig, psd_mod, psd_rec]
            labels = ["Señal Original", "Señal Modulada", "Señal Recuperada"]
            
            fig_psd = create_multiband_psd_plot(frequencies, psds, labels, 
                                              "Comparación Espectral DSB-SC")
            
            # Add carrier frequency markers
            fig_psd.add_vline(x=fc/1000, line_dash="dash", line_color="red", 
                             annotation_text=f"fc = {fc/1000:.1f} kHz")
            fig_psd.add_vline(x=-fc/1000, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_psd, use_container_width=True)
            
            # Spectral analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Observaciones espectrales:**")
                
                # Find signal bandwidth
                signal_bw = estimate_signal_bandwidth(f_orig, psd_orig)
                st.text(f"• Ancho de banda señal: {signal_bw/1000:.1f} kHz")
                st.text(f"• Bandas laterales: {(fc-signal_bw)/1000:.1f} - {(fc+signal_bw)/1000:.1f} kHz")
                st.text(f"• Eficiencia espectral: {2*signal_bw/fs_actual*100:.1f}%")
            
            with col2:
                st.markdown("**⚙️ Parámetros del filtro:**")
                st.text(f"• Frecuencia de corte: {cutoff_ratio*fc/1000:.1f} kHz")
                st.text(f"• Orden: {filter_order}")
                st.text(f"• Ventana: {filter_window}")
                
                # Filter performance
                if cutoff_ratio * fc > signal_bw * 1.5:
                    st.success("✅ Filtro bien dimensionado")
                else:
                    st.warning("⚠️ Filtro podría atenuar la señal")
    
    with tab3:
        st.subheader("Respuesta en frecuencia del filtro receptor")
        
        # Get filter response
        f_filter, h_mag, h_phase = cached_filter_response(
            hash_array(sos), fs_actual, 8192, _filter_coeffs=sos
        )
        
        # Create filter response plot
        fig_filter = create_filter_response_plot(f_filter, h_mag, h_phase)
        
        # Add design specifications
        cutoff_freq = cutoff_ratio * fc
        fig_filter.add_vline(x=cutoff_freq/1000, line_dash="dot", line_color="green",
                           annotation_text=f"Corte: {cutoff_freq/1000:.1f} kHz")
        
        st.plotly_chart(fig_filter, use_container_width=True)
        
        # Filter analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Find -3dB point
            idx_3db = np.where(h_mag <= -3)[0]
            if len(idx_3db) > 0:
                f_3db = f_filter[idx_3db[0]]
                st.metric("📉 Frecuencia -3dB", f"{f_3db/1000:.2f} kHz")
            
            # Find stopband attenuation
            stopband_start = cutoff_freq * 2
            if stopband_start < fs_actual/2:
                idx_stop = np.where(f_filter >= stopband_start)[0]
                if len(idx_stop) > 0:
                    stopband_atten = np.min(h_mag[idx_stop])
                    st.metric("🚫 Atenuación banda eliminada", f"{stopband_atten:.1f} dB")
        
        with col2:
            # Group delay estimate
            passband_idx = f_filter <= cutoff_freq
            if np.any(passband_idx):
                avg_phase = np.mean(h_phase[passband_idx])
                group_delay_est = -avg_phase / (360 * np.mean(f_filter[passband_idx])) * 1000
                st.metric("⏱️ Retardo estimado", f"{abs(group_delay_est):.2f} ms")
            
            # Filter quality factor
            transition_bw = f_3db - cutoff_freq if 'f_3db' in locals() else 0
            if transition_bw > 0:
                q_factor = cutoff_freq / transition_bw
                st.metric("🎯 Factor Q", f"{q_factor:.1f}")
    
    with tab4:
        st.subheader("🎵 Reproducción y comparación de audio")
        
        # Audio comparison widget
        create_audio_comparison_widget(
            x, x_recovered, fs_actual, 
            ("Audio Original", "Audio Recuperado")
        )
        
        # Audio analysis
        if len(x) == len(x_recovered):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Análisis de calidad de audio:**")
                
                # THD estimation
                if analysis_opts['show_metrics']:
                    try:
                        # Find dominant frequency for THD calculation
                        f_dom = find_dominant_frequency(x, fs_actual)
                        if f_dom > 0:
                            sinad, snr_audio, thd = compute_sinad(x_recovered, fs_actual, f_dom)
                            st.metric("🎵 SINAD", f"{sinad:.1f} dB")
                            st.metric("🎶 THD", f"{thd:.1f} dB")
                    except:
                        pass
                
                # Dynamic range
                dynamic_range = compute_papr(x_recovered)
                st.metric("📈 Rango dinámico", f"{dynamic_range:.1f} dB")
            
            with col2:
                st.markdown("**🎛️ Control de efectos de prueba:**")
                
                # Simple audio effects for testing
                x_with_effects = create_audio_effects_panel(x_recovered, fs_actual)
                
                if not np.array_equal(x_with_effects, x_recovered):
                    st.markdown("**Audio con efectos:**")
                    create_audio_player(x_with_effects, fs_actual, "effects_audio")
    
    # Summary and recommendations
    st.markdown("---")
    st.subheader("📋 Resumen y Recomendaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Resultados obtenidos:**")
        st.markdown(f"• SNR: **{snr:.1f} dB** ({'Excelente' if snr > 40 else 'Bueno' if snr > 20 else 'Regular'})")
        st.markdown(f"• Correlación: **{correlation:.3f}** ({'Alta' if correlation > 0.95 else 'Media' if correlation > 0.8 else 'Baja'})")
        st.markdown(f"• Eficiencia del filtro: **{cutoff_ratio:.1f}×fc**")
        st.markdown(f"• Complejidad: **{filter_order} taps**")
    
    with col2:
        st.markdown("**💡 Recomendaciones:**")
        
        recommendations = []
        
        if snr < 30:
            recommendations.append("• Aumentar orden del filtro para mejor SNR")
        
        if correlation < 0.9:
            recommendations.append("• Verificar sincronización de portadora")
        
        if cutoff_ratio < 0.5:
            recommendations.append("• Aumentar frecuencia de corte del filtro")
        elif cutoff_ratio > 1.5:
            recommendations.append("• Reducir frecuencia de corte para mejor selectividad")
        
        if fc < 5 * estimate_signal_bandwidth(f_orig, psd_orig):
            recommendations.append("• Aumentar frecuencia portadora para mejor separación")
        
        if not recommendations:
            recommendations.append("• ✅ Configuración óptima alcanzada")
        
        for rec in recommendations:
            st.markdown(rec)
    
    # Export and download options
    st.markdown("---")
    st.subheader("💾 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Exportar todas las señales"):
            # Prepare export data
            export_data = {
                'original_signal': x,
                'modulated_signal': y_modulated,
                'recovered_signal': x_recovered,
                'time_vector': np.arange(len(x)) / fs_actual,
                'sample_rate': fs_actual,
                'carrier_frequency': fc,
                'filter_order': filter_order,
                'snr_db': snr,
                'correlation': correlation
            }
            
            # Create download
            import io
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **export_data)
            
            st.download_button(
                label="⬇️ Descargar archivo NPZ",
                data=buffer.getvalue(),
                file_name=f"dsb_sc_results_fc{fc/1000:.0f}kHz.npz",
                mime="application/octet-stream"
            )
    
    with col2:
        if st.button("🎵 Exportar audio procesado"):
            # Export recovered audio
            wav_bytes = audio_to_wav_bytes(x_recovered, fs_actual)
            
            st.download_button(
                label="⬇️ Descargar WAV",
                data=wav_bytes,
                file_name=f"dsb_sc_recovered_fc{fc/1000:.0f}kHz.wav",
                mime="audio/wav"
            )
    
    with col3:
        if st.button("📊 Exportar métricas"):
            # Create metrics report
            import pandas as pd
            
            metrics_df = pd.DataFrame({
                'Parámetro': ['SNR (dB)', 'Correlación', 'Error RMS', 'Frecuencia portadora (kHz)', 
                             'Frecuencia muestreo (kHz)', 'Orden filtro', 'Frecuencia corte (kHz)'],
                'Valor': [snr, correlation, rms_error, fc/1000, fs_actual/1000, 
                         filter_order, cutoff_ratio*fc/1000]
            })
            
            csv_metrics = metrics_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV",
                data=csv_metrics,
                file_name=f"dsb_sc_metrics_fc{fc/1000:.0f}kHz.csv",
                mime="text/csv"
            )
    
    # Technical information panel
    with st.expander("🔬 Información técnica detallada"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📡 Parámetros del sistema:**")
            st.code(f"""
Modulación: DSB-SC
Portadora: {fc/1000:.1f} kHz
Muestreo: {fs_actual/1000:.1f} kHz
Relación fc/fs: {fc/fs_actual:.3f}
Orden filtro: {filter_order}
Ventana: {filter_window}
Corte filtro: {cutoff_ratio:.1f} × fc
            """)
        
        with col2:
            st.markdown("**📊 Estadísticas de la señal:**")
            signal_stats = signal_statistics(x)
            recovered_stats = signal_statistics(x_recovered)
            
            st.code(f"""
Original:
  RMS: {signal_stats['rms']:.4f}
  Pico: {signal_stats['max']:.4f}
  Rango: {signal_stats['peak_to_peak']:.4f}

Recuperada:
  RMS: {recovered_stats['rms']:.4f}
  Pico: {recovered_stats['max']:.4f}
  Rango: {recovered_stats['peak_to_peak']:.4f}
            """)
    
    # Theoretical explanation
    with st.expander("📚 Fundamentos de DSB-SC"):
        st.markdown("""
        ### 📡 Modulación DSB-SC (Double Sideband Suppressed Carrier)
        
        **Principio de funcionamiento:**
        
        La modulación DSB-SC multiplica la señal mensaje x(t) por una portadora cos(ωct):
        
        ```
        y(t) = x(t) × cos(ωct)
        ```
        
        **En el dominio de la frecuencia:**
        ```
        Y(f) = ½[X(f-fc) + X(f+fc)]
        ```
        
        **Ventajas:**
        - ✅ No desperdicia potencia en portadora (100% eficiencia de potencia)
        - ✅ Ancho de banda mínimo (2B, donde B es el ancho de banda del mensaje)
        - ✅ Buena calidad para señales de audio
        
        **Desventajas:**
        - ❌ Requiere demodulación coherente (sincronización exacta)
        - ❌ Sensible a errores de fase y frecuencia
        - ❌ Mayor complejidad en el receptor
        
        **Demodulación coherente:**
        
        1. **Mezclado:** Multiplicar por 2cos(ωct)
           ```
           2y(t)cos(ωct) = 2x(t)cos²(ωct) = x(t)[1 + cos(2ωct)]
           ```
        
        2. **Filtrado:** Eliminar componente en 2fc con filtro pasa-bajas
           ```
           x̂(t) = LPF{x(t)[1 + cos(2ωct)]} = x(t)
           ```
        
        **Consideraciones de diseño:**
        
        - **Frecuencia portadora:** fc >> B (típicamente fc ≥ 5B)
        - **Filtro receptor:** Corte entre B y 2fc-B
        - **Sincronización:** Error de fase < 5° para buena calidad
        - **Muestreo:** fs ≥ 2(fc + B) para evitar aliasing
        
        **Aplicaciones:**
        - Comunicaciones de radio AM profesional
        - Sistemas de transmisión punto a punto
        - Modulación en comunicaciones digitales (BPSK, QPSK)
        """)

# Helper functions for this page

def generate_synthetic_voice(duration: float, fs: float, f0: float, formants: list) -> Tuple[np.ndarray, float]:
    """Generate synthetic voice-like signal with specified formants."""
    t = np.arange(0, duration, 1/fs)
    
    # Fundamental frequency with slight vibrato
    vibrato_freq = 5  # Hz
    vibrato_depth = 0.02
    f_inst = f0 * (1 + vibrato_depth * np.sin(2*np.pi*vibrato_freq*t))
    
    # Generate harmonic series
    signal = np.zeros_like(t)
    
    # Add harmonics with formant shaping
    for harmonic in range(1, 20):
        freq = harmonic * f_inst
        amplitude = 1 / harmonic  # Natural harmonic rolloff
        
        # Boost amplitude near formants
        for formant in formants:
            if abs(freq.mean() - formant) < formant * 0.3:
                amplitude *= 3
        
        # Add harmonic with phase noise for naturalness
        phase_noise = 0.1 * np.random.randn(len(t))
        signal += amplitude * np.sin(2*np.pi*freq*t + phase_noise)
    
    # Apply envelope (speech-like)
    envelope = np.ones_like(t)
    # Add pauses
    pause_locations = np.arange(0.5, duration, 1.5)  # Pause every 1.5 seconds
    for pause_time in pause_locations:
        pause_start = int(pause_time * fs)
        pause_end = int((pause_time + 0.2) * fs)
        if pause_end < len(envelope):
            envelope[pause_start:pause_end] *= np.linspace(1, 0.1, pause_end-pause_start)
    
    signal *= envelope
    
    # Normalize
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32), fs

def estimate_signal_bandwidth(frequencies: np.ndarray, psd: np.ndarray, threshold_db: float = -20) -> float:
    """Estimate signal bandwidth from PSD."""
    # Find frequencies where PSD is above threshold
    peak_psd = np.max(psd)
    mask = psd >= (peak_psd + threshold_db)
    
    if np.any(mask):
        valid_freqs = frequencies[mask]
        return np.max(valid_freqs) - np.min(valid_freqs)
    else:
        return 0.0

def find_dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Find dominant frequency in signal for analysis."""
    # Simple peak finding in spectrum
    f, psd = compute_psd(signal, fs, 'periodogram')
    
    # Find peak (excluding DC)
    if len(f) > 1:
        peak_idx = np.argmax(psd[1:]) + 1
        return f[peak_idx]
    else:
        return 0.0

if __name__ == "__main__":
    main()