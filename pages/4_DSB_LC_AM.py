"""
DSB-LC (Double Sideband Large Carrier) AM Page

Interactive demonstration of AM modulation with large carrier,
including overmodulation effects and envelope detection.
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

st.set_page_config(page_title="DSB-LC AM", page_icon="📻", layout="wide")

def main():
    st.title("📻 Modulación DSB-LC (AM con Portadora Grande)")
    
    st.markdown("""
    **Principio:** La modulación AM añade una portadora grande para permitir detección de envolvente simple.
    
    **Ecuación:** s(t) = [1 + μm(t)] × cos(ωct)
    
    **Índice de modulación:** μ = (Emax - Emin) / (Emax + Emin)
    """)
    
    # Sidebar controls
    create_sidebar_header("Configuración AM")
    
    # Message signal composition
    create_parameter_section("Señal Mensaje")
    
    # Choice between pre-defined multi-tone or custom
    signal_mode = st.sidebar.selectbox(
        "🎵 Modo de señal:",
        ["Triple tono (A + B + C)", "Audio personalizado", "Forma de onda sintética"],
        help="Selecciona el tipo de señal mensaje"
    )
    
    if signal_mode == "Triple tono (A + B + C)":
        st.sidebar.markdown("**Componentes del triple tono:**")
        
        # Amplitude controls
        A = st.sidebar.slider("🎶 Amplitud A:", 0.0, 1.0, 0.5, 0.05)
        B = st.sidebar.slider("🎶 Amplitud B:", 0.0, 1.0, 0.3, 0.05) 
        C = st.sidebar.slider("🎶 Amplitud C:", 0.0, 1.0, 0.2, 0.05)
        
        # Frequency controls
        f1 = st.sidebar.number_input("🎵 Frecuencia f1 (Hz):", 100, 2000, 500, 50)
        f2 = st.sidebar.number_input("🎵 Frecuencia f2 (Hz):", 100, 2000, 1000, 50)
        f3 = st.sidebar.number_input("🎵 Frecuencia f3 (Hz):", 100, 2000, 1500, 50)
        
        duration = st.sidebar.slider("⏱️ Duración (s):", 1, 10, 5)
        
    elif signal_mode == "Audio personalizado":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Subir archivo de audio:",
            type=['wav', 'mp3', 'flac'],
            help="Archivo será normalizado automáticamente"
        )
        
    else:  # Synthetic waveform
        waveform_type = st.sidebar.selectbox(
            "📊 Tipo de forma de onda:",
            ["Triangular", "Cuadrada", "Diente de sierra", "Senoidal modulada"],
            help="Formas de onda para análisis AM"
        )
        
        wave_freq = st.sidebar.number_input("🎵 Frecuencia (Hz):", 10, 1000, 100, 10)
        duration = st.sidebar.slider("⏱️ Duración (s):", 1, 10, 5)
    
    # Modulation parameters
    create_parameter_section("Parámetros AM")
    
    # Modulation index selection
    mu_preset = st.sidebar.selectbox(
        "📐 Índice de modulación (μ):",
        ["μ = 0.7 (Submodulación)", "μ = 1.0 (100% modulación)", "μ = 1.2 (Sobremodulación)", "Personalizado"],
        index=1,
        help="μ > 1 causa distorsión por sobremodulación"
    )
    
    if mu_preset == "Personalizado":
        mu = st.sidebar.slider("🎛️ μ personalizado:", 0.1, 2.0, 1.0, 0.1)
    else:
        mu_values = {"μ = 0.7 (Submodulación)": 0.7, "μ = 1.0 (100% modulación)": 1.0, "μ = 1.2 (Sobremodulación)": 1.2}
        mu = mu_values[mu_preset]
    
    # RF parameters
    mod_params = create_modulation_controls()
    fc = mod_params['fc']
    fs = mod_params['fs']
    
    # Envelope detection parameters
    create_parameter_section("Detector de Envolvente")
    
    detection_method = st.sidebar.selectbox(
        "🔧 Método de detección:",
        ["Rectificador + Filtro", "Transformada de Hilbert", "Detector de pico"],
        help="Método para extraer la envolvente"
    )
    
    if detection_method == "Rectificador + Filtro":
        env_filter_order = st.sidebar.slider("📊 Orden filtro envolvente:", 2, 10, 4)
        env_cutoff_ratio = st.sidebar.slider("🔧 Corte relativo (×fc):", 0.01, 0.2, 0.05, 0.01)
    
    # Analysis options
    analysis_opts = create_analysis_controls()
    
    # Generate message signal
    with st.spinner("Generando señal mensaje..."):
        if signal_mode == "Triple tono (A + B + C)":
            # Generate triple tone signal
            t = np.arange(0, duration, 1/fs)
            m = (A * np.sin(2*np.pi*f1*t) + 
                 B * np.sin(2*np.pi*f2*t) + 
                 C * np.sin(2*np.pi*f3*t))
            
            # Normalize to ±1 range
            if np.max(np.abs(m)) > 0:
                m = m / np.max(np.abs(m))
            
        elif signal_mode == "Audio personalizado":
            if uploaded_file is not None:
                m, fs_temp = load_and_preprocess_audio(uploaded_file, target_fs=None, normalize=True)
                if fs_temp != fs:
                    m = resample_signal(m, fs_temp, fs)
            else:
                st.warning("⚠️ Sube un archivo de audio para continuar")
                st.stop()
                
        else:  # Synthetic waveform
            t = np.arange(0, duration, 1/fs)
            
            if waveform_type == "Triangular":
                m = 2 * np.abs(2 * (wave_freq * t - np.floor(wave_freq * t + 0.5))) - 1
            elif waveform_type == "Cuadrada":
                m = np.sign(np.sin(2*np.pi*wave_freq*t))
            elif waveform_type == "Diente de sierra":
                m = 2 * (wave_freq * t - np.floor(wave_freq * t + 0.5))
            else:  # Senoidal modulada
                carrier_env = np.sin(2*np.pi*wave_freq*t)
                modulation_env = 0.5 * np.sin(2*np.pi*wave_freq/10*t)
                m = carrier_env * (1 + modulation_env)
                m = m / np.max(np.abs(m))  # Normalize
    
    # Validate signal
    if not validate_audio_parameters(m, fs, fc):
        st.stop()
    
    # Create processing pipeline
    am_stages = [
        "Mensaje m(t)", 
        "Suma DC [1+μm(t)]", 
        "Modulador AM", 
        "Canal s(t)", 
        "Detector envolvente", 
        "Mensaje m̂(t)"
    ]
    create_processing_pipeline_diagram(am_stages)
    
    # Perform AM modulation
    st.subheader("🔄 Procesamiento AM")
    
    with st.spinner("Procesando modulación AM..."):
        # AM modulation
        s_am = dsb_lc_modulate(m, mu, fc, fs)
        
        # Envelope detection
        if detection_method == "Rectificador + Filtro":
            envelope_detected = envelope_detector(s_am, fs, fc)
        elif detection_method == "Transformada de Hilbert":
            envelope_detected = hilbert_envelope_detector(s_am)
        else:  # Peak detector
            envelope_detected = peak_envelope_detector(s_am, fs, fc)
        
        # Remove DC component and recover message
        dc_level = np.mean(envelope_detected)
        m_recovered = (envelope_detected - dc_level) / mu if mu > 0 else envelope_detected - dc_level
        
        # Estimate modulation index from detected envelope
        mu_estimated = compute_modulation_index(envelope_detected)
    
    # Compute quality metrics
    with st.spinner("Analizando calidad AM..."):
        # Limit comparison to same length
        min_len = min(len(m), len(m_recovered))
        m_comp = m[:min_len]
        m_rec_comp = m_recovered[:min_len]
        
        metrics = cached_metrics_computation(
            hash_array(m_comp), hash_array(m_rec_comp), 
            _signal1=m_comp, _signal2=m_rec_comp
        )
        
        snr = metrics['snr']
        correlation = metrics['correlation']
        rms_error = metrics['rms_error']
        
        # Distortion analysis for overmodulation
        if mu > 1.0:
            distortion_pct = analyze_overmodulation_distortion(s_am, mu, fc, fs)
        else:
            distortion_pct = 0.0
    
    # Display metrics dashboard
    st.subheader("📊 Métricas de Modulación AM")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📐 μ configurado", f"{mu:.2f}")
    with col2:
        st.metric("📏 μ estimado", f"{mu_estimated:.2f}")
    with col3:
        st.metric("📊 SNR", f"{snr:.1f} dB")
    with col4:
        st.metric("🔗 Correlación", f"{correlation:.4f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("❌ Error RMS", f"{rms_error:.6f}")
    with col2:
        efficiency = mu**2 / (2 + mu**2) * 100  # AM efficiency formula
        st.metric("⚡ Eficiencia", f"{efficiency:.1f}%")
    with col3:
        if mu > 1.0:
            st.metric("⚠️ Distorsión", f"{distortion_pct:.2f}%", delta="Sobremodulación")
        else:
            st.metric("✅ Distorsión", f"{distortion_pct:.2f}%")
    with col4:
        bandwidth_usage = estimate_am_bandwidth(m, fs) / (fs/2) * 100
        st.metric("📏 Uso espectral", f"{bandwidth_usage:.1f}%")
    
    # Quality assessment alerts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if mu > 1.0:
            st.error("⚠️ SOBREMODULACIÓN DETECTADA")
            st.markdown("La envolvente se invierte causando distorsión")
        elif mu > 0.9:
            st.warning("⚠️ Cerca de sobremodulación")
        else:
            st.success("✅ Modulación normal")
    
    with col2:
        if abs(mu_estimated - mu) < 0.1:
            st.success("✅ μ bien detectado")
        else:
            st.warning(f"⚠️ Error en estimación μ: {abs(mu_estimated - mu):.2f}")
    
    with col3:
        if snr > 30:
            st.success("✅ Excelente calidad")
        elif snr > 15:
            st.info("ℹ️ Buena calidad")
        else:
            st.warning("⚠️ Calidad regular")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕒 Señales Temporales", 
        "📊 Análisis Espectral", 
        "📈 Detección de Envolvente",
        "⚠️ Análisis de Distorsión",
        "🎵 Audio"
    ])
    
    # Reemplazar la sección with tab1: en tu archivo DSB-LC por este código:

    with tab1:
        st.subheader("Formas de onda del sistema AM")
        
        # 🎛️ CONTROLES DE VISUALIZACIÓN MEJORADOS
        st.markdown("**🎛️ Controles de visualización:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Control de ciclos de portadora
            if signal_mode == "Triple tono (A + B + C)":
                # Para multi-tono, calcular el período de la señal más lenta
                min_freq = min(f1, f2, f3)
                signal_period = 1 / min_freq  # Período de la componente más lenta
                
                # Calcular cuántos períodos de portadora hay en un período de señal
                carrier_periods_per_signal = fc * signal_period
                
                cycle_options = [
                    "1 período señal",
                    "2 períodos señal", 
                    "5 períodos señal",
                    "10 períodos señal",
                    "1 período portadora",
                    "5 períodos portadora",
                    "10 períodos portadora",
                    "50 períodos portadora",
                    "Todos"
                ]
            else:
                # Para otras señales, usar frecuencia base estimada
                signal_period = 1 / wave_freq if 'wave_freq' in locals() else 0.01
                cycle_options = [
                    "1 ciclo señal",
                    "2 ciclos señal", 
                    "5 ciclos señal",
                    "10 ciclos señal",
                    "1 ciclo portadora",
                    "5 ciclos portadora",
                    "10 ciclos portadora",
                    "50 ciclos portadora",
                    "Todos"
                ]
            
            view_mode = st.selectbox(
                "🔍 Vista temporal:",
                cycle_options,
                index=2,  # Default: 5 períodos
                key="view_cycles"
            )
        
        with col2:
            # Control de unidades
            time_unit = st.selectbox(
                "⏱️ Unidad tiempo:",
                ["Milisegundos", "Segundos"],
                index=0,
                key="time_unit_am"
            )
        
        with col3:
            # Control de zoom fino
            zoom_factor = st.slider(
                "🔎 Zoom adicional:",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="zoom_factor_am"
            )
        
        with col4:
            # Control de posición
            if view_mode != "Todos":
                position_pct = st.slider(
                    "📍 Posición (%):",
                    min_value=0,
                    max_value=90,
                    value=0,
                    step=5,
                    key="position_am"
                )
            else:
                position_pct = 0
                st.metric("📏 Duración total", f"{len(m)/fs:.2f} s")
        
        # Calcular parámetros de visualización
        total_duration = len(m) / fs
        
        if view_mode == "Todos":
            display_duration = total_duration
            start_time = 0
        
        elif "portadora" in view_mode:
            # Extraer número de períodos de portadora
            num_periods = int(view_mode.split()[0])
            carrier_period = 1 / fc
            display_duration = num_periods * carrier_period / zoom_factor
            max_start = max(0, total_duration - display_duration)
            start_time = (position_pct / 100) * max_start
        
        elif "señal" in view_mode or "ciclo" in view_mode:
            # Extraer número de períodos de señal
            num_periods = int(view_mode.split()[0])
            display_duration = num_periods * signal_period / zoom_factor
            max_start = max(0, total_duration - display_duration)
            start_time = (position_pct / 100) * max_start
        
        # Asegurar que no excedemos los límites
        if start_time + display_duration > total_duration:
            display_duration = total_duration - start_time
        
        # Convertir a muestras
        start_sample = int(start_time * fs)
        n_display = int(display_duration * fs)
        end_sample = min(start_sample + n_display, len(m))
        
        # Vector temporal para visualización
        if time_unit == "Milisegundos":
            t_display = np.arange(start_sample, end_sample) / fs * 1000
            time_label = "Tiempo (ms)"
            time_suffix = "ms"
        else:
            t_display = np.arange(start_sample, end_sample) / fs
            time_label = "Tiempo (s)"
            time_suffix = "s"
        
        # 📊 INFORMACIÓN DE LA VISTA ACTUAL
        st.markdown("**📊 Información de la vista actual:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Muestras", f"{end_sample - start_sample:,}")
        with col2:
            if time_unit == "Milisegundos":
                st.metric("⏱️ Duración", f"{display_duration*1000:.1f} ms")
            else:
                st.metric("⏱️ Duración", f"{display_duration:.3f} s")
        with col3:
            # Calcular número de ciclos mostrados
            if display_duration > 0:
                if "portadora" in view_mode:
                    cycles_shown = display_duration * fc
                    st.metric("🔄 Ciclos portadora", f"{cycles_shown:.1f}")
                else:
                    cycles_shown = display_duration / signal_period
                    st.metric("🔄 Ciclos señal", f"{cycles_shown:.1f}")
        with col4:
            progress_pct = (start_time + display_duration/2) / total_duration * 100
            st.metric("📍 Posición", f"{progress_pct:.1f}%")
        
        # 🎨 GRÁFICAS PRINCIPALES
        # Create comprehensive time domain plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f"Señal mensaje m(t) - {signal_mode}",
                f"Señal AM s(t) = [1 + {mu:.1f}m(t)]cos(ωct)",
                f"Envolvente detectada (método: {detection_method})",
                "Mensaje recuperado m̂(t) vs original"
            ],
            vertical_spacing=0.08
        )
        
        # Extraer datos para la ventana actual
        m_window = m[start_sample:end_sample]
        s_am_window = s_am[start_sample:end_sample]
        env_window = envelope_detected[start_sample:end_sample]
        m_rec_window = m_recovered[start_sample:end_sample]
        
        # Original message signal
        fig.add_trace(go.Scatter(
            x=t_display, y=m_window,
            mode='lines', name='m(t)', line=dict(color='blue', width=2),
            hovertemplate=f'Mensaje<br>t: %{{x:.2f}} {time_suffix}<br>m(t): %{{y:.4f}}<extra></extra>'
        ), row=1, col=1)
        
        # AM signal
        fig.add_trace(go.Scatter(
            x=t_display, y=s_am_window,
            mode='lines', name='s(t)', line=dict(color='red', width=1),
            hovertemplate=f'AM<br>t: %{{x:.2f}} {time_suffix}<br>s(t): %{{y:.4f}}<extra></extra>'
        ), row=2, col=1)
        
        # Envelope
        fig.add_trace(go.Scatter(
            x=t_display, y=env_window,
            mode='lines', name='Envolvente', line=dict(color='green', width=3),
            hovertemplate=f'Envolvente<br>t: %{{x:.2f}} {time_suffix}<br>Env: %{{y:.4f}}<extra></extra>'
        ), row=3, col=1)
        
        # Add theoretical envelope for comparison
        theoretical_envelope = 1 + mu * m_window
        fig.add_trace(go.Scatter(
            x=t_display, y=theoretical_envelope,
            mode='lines', name='Envolvente teórica', 
            line=dict(color='lightgreen', width=2, dash='dash'),
            hovertemplate=f'Teórica<br>t: %{{x:.2f}} {time_suffix}<br>Env: %{{y:.4f}}<extra></extra>'
        ), row=3, col=1)
        
        # Recovered vs original message
        fig.add_trace(go.Scatter(
            x=t_display, y=m_window,
            mode='lines', name='Original', line=dict(color='blue', width=2),
            hovertemplate=f'Original<br>t: %{{x:.2f}} {time_suffix}<br>m(t): %{{y:.4f}}<extra></extra>'
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=t_display, y=m_rec_window,
            mode='lines', name='Recuperado', 
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate=f'Recuperado<br>t: %{{x:.2f}} {time_suffix}<br>m̂(t): %{{y:.4f}}<extra></extra>'
        ), row=4, col=1)
        
        fig.update_xaxes(title_text=time_label, row=4, col=1)
        fig.update_yaxes(title_text="Amplitud")
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 🔍 VISTA DETALLADA ADICIONAL
        if st.checkbox("🔬 Vista ultra-detallada (alta resolución)"):
            st.markdown("**🔎 Análisis de alta resolución:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                detail_duration_ms = st.slider(
                    "Ventana detalle (ms):", 
                    min_value=1, 
                    max_value=50, 
                    value=10,
                    key="detail_window"
                )
            
            with col2:
                detail_start_ms = st.slider(
                    "Inicio detalle (ms):", 
                    min_value=0, 
                    max_value=int(display_duration*1000 - detail_duration_ms), 
                    value=0,
                    key="detail_start"
                )
            
            # Calcular ventana de detalle
            detail_start_sample = start_sample + int(detail_start_ms/1000 * fs)
            detail_n_samples = int(detail_duration_ms/1000 * fs)
            detail_end_sample = min(detail_start_sample + detail_n_samples, end_sample)
            
            if detail_end_sample > detail_start_sample:
                t_detail = np.arange(detail_start_sample, detail_end_sample) / fs * 1000
                
                # Crear gráfica de envolvente detallada
                fig_detail = create_am_envelope_plot(
                    t_detail/1000,  # Convertir de ms a s para la función
                    s_am[detail_start_sample:detail_end_sample],
                    envelope_detected[detail_start_sample:detail_end_sample],
                    mu
                )
                
                # Agregar señal mensaje para referencia
                fig_detail.add_trace(go.Scatter(
                    x=t_detail, y=1 + mu * m[detail_start_sample:detail_end_sample],
                    mode='lines', name='Envolvente ideal',
                    line=dict(color='black', width=2, dash='dot'),
                    hovertemplate='Ideal<br>t: %{x:.2f} ms<br>Ideal: %{y:.4f}<extra></extra>'
                ))
                
                fig_detail.update_layout(
                    title=f"Vista Detallada: {detail_duration_ms}ms desde {detail_start_ms}ms",
                    height=500
                )
                
                st.plotly_chart(fig_detail, use_container_width=True)
                
                # Métricas de la ventana detallada
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    detail_cycles_carrier = detail_duration_ms/1000 * fc
                    st.metric("🔄 Ciclos portadora", f"{detail_cycles_carrier:.1f}")
                
                with col2:
                    if signal_period > 0:
                        detail_cycles_signal = detail_duration_ms/1000 / signal_period
                        st.metric("🔄 Ciclos señal", f"{detail_cycles_signal:.2f}")
                
                with col3:
                    # Calcular distorsión local si hay sobremodulación
                    if mu > 1.0:
                        env_local = envelope_detected[detail_start_sample:detail_end_sample]
                        inversions_local = np.sum(env_local < np.mean(env_local)*0.5)
                        inversion_pct_local = inversions_local / len(env_local) * 100
                        st.metric("⚠️ Inversiones (%)", f"{inversion_pct_local:.1f}")
                    else:
                        st.metric("✅ Estado", "Normal")
        
        # ⚡ NAVEGACIÓN RÁPIDA
        if view_mode != "Todos":
            st.markdown("**⚡ Navegación rápida:**")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("⏮️ Inicio", key="nav_start_am"):
                    st.session_state.position_am = 0
                    st.rerun()
            
            with col2:
                if st.button("⬅️ Anterior", key="nav_prev_am"):
                    new_pos = max(0, position_pct - 10)
                    st.session_state.position_am = new_pos
                    st.rerun()
            
            with col3:
                if st.button("🎯 Centro", key="nav_center_am"):
                    st.session_state.position_am = 45
                    st.rerun()
            
            with col4:
                if st.button("➡️ Siguiente", key="nav_next_am"):
                    new_pos = min(90, position_pct + 10)
                    st.session_state.position_am = new_pos
                    st.rerun()
            
            with col5:
                if st.button("⏭️ Final", key="nav_end_am"):
                    st.session_state.position_am = 90
                    st.rerun()
        
        # 📋 PRESETS DE VISUALIZACIÓN
        with st.expander("📋 Presets de visualización recomendados"):
            st.markdown("""
            **🎯 Presets útiles según el análisis:**
            
            • **Para ver modulación:** 2-5 períodos señal + zoom 1.0
            • **Para ver portadora:** 10-20 períodos portadora + zoom 2.0  
            • **Para ver sobremodulación:** 1 período señal + zoom 3.0
            • **Para análisis general:** 5 períodos señal + zoom 1.0
            • **Para precisión:** Vista ultra-detallada con 5-10ms
            """)
            
            # Botones de presets
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🎵 Preset Modulación"):
                    st.session_state.view_cycles = "2 períodos señal" if "señal" in cycle_options[1] else "2 ciclos señal"
                    st.session_state.zoom_factor_am = 1.0
                    st.rerun()
            
            with col2:
                if st.button("📡 Preset Portadora"):
                    st.session_state.view_cycles = "10 períodos portadora"
                    st.session_state.zoom_factor_am = 2.0
                    st.rerun()
            
            with col3:
                if st.button("⚠️ Preset Distorsión"):
                    st.session_state.view_cycles = "1 período señal" if "señal" in cycle_options[0] else "1 ciclo señal"
                    st.session_state.zoom_factor_am = 3.0
                    st.rerun()
            
            with col4:
                if st.button("🔄 Reset a Defecto"):
                    st.session_state.view_cycles = cycle_options[2]  # 5 períodos
                    st.session_state.zoom_factor_am = 1.0
                    st.session_state.position_am = 0
                    st.rerun()
    
    with tab2:
        st.subheader("Análisis espectral del sistema AM")
        
        if analysis_opts['show_spectrum']:
            # Compute PSDs
            f_msg, psd_msg = cached_psd_computation(
                hash_array(m), fs, analysis_opts['psd_method'], _signal=m
            )
            f_am, psd_am = cached_psd_computation(
                hash_array(s_am), fs, analysis_opts['psd_method'], _signal=s_am
            )
            f_rec, psd_rec = cached_psd_computation(
                hash_array(m_recovered), fs, analysis_opts['psd_method'], _signal=m_recovered
            )
            
            # Create spectral plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Espectro del mensaje:**")
                fig_msg = create_psd_plot(f_msg, psd_msg, "Señal Mensaje", 'blue')
                st.plotly_chart(fig_msg, use_container_width=True)
                
                st.markdown("**📊 Mensaje recuperado:**")
                fig_rec = create_psd_plot(f_rec, psd_rec, "Mensaje Recuperado", 'orange')
                st.plotly_chart(fig_rec, use_container_width=True)
            
            with col2:
                st.markdown("**📡 Espectro AM:**")
                fig_am = create_psd_plot(f_am, psd_am, "Señal AM Modulada", 'red')
                
                # Add carrier and sideband markers
                fig_am.add_vline(x=fc/1000, line_dash="solid", line_color="black", 
                               annotation_text=f"Portadora {fc/1000:.1f} kHz")
                
                # Estimate message bandwidth for sideband markers
                msg_bw = estimate_am_bandwidth(m, fs)
                fig_am.add_vline(x=(fc-msg_bw)/1000, line_dash="dot", line_color="gray", 
                               annotation_text="LSB")
                fig_am.add_vline(x=(fc+msg_bw)/1000, line_dash="dot", line_color="gray", 
                               annotation_text="USB")
                
                st.plotly_chart(fig_am, use_container_width=True)
            
            # Spectral analysis
            st.markdown("**🎯 Análisis de bandas laterales:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📏 Ancho banda mensaje", f"{msg_bw/1000:.2f} kHz")
                st.metric("📊 Ancho banda AM", f"{2*msg_bw/1000:.2f} kHz")
            
            with col2:
                # Power distribution analysis
                carrier_power_ratio = 1 / (2 + mu**2)
                sideband_power_ratio = mu**2 / (2 * (2 + mu**2))
                
                st.metric("⚡ Potencia portadora", f"{carrier_power_ratio*100:.1f}%")
                st.metric("📡 Potencia bandas lat.", f"{sideband_power_ratio*2*100:.1f}%")
            
            with col3:
                # Spectral efficiency
                info_power_ratio = mu**2 / (2 + mu**2)
                spectral_eff = info_power_ratio * 100
                
                st.metric("📈 Eficiencia espectral", f"{spectral_eff:.1f}%")
                
                if mu < 0.5:
                    st.warning("⚠️ Baja eficiencia")
                elif mu > 1.0:
                    st.error("⚠️ Sobremodulación")
                else:
                    st.success("✅ Eficiencia aceptable")
    
    with tab3:
        st.subheader("Análisis de detección de envolvente")
        
        # Envelope detection comparison
        st.markdown(f"**🔧 Método usado:** {detection_method}")
        
        # Compare different detection methods
        if st.checkbox("📊 Comparar métodos de detección"):
            # Compute all methods
            env_rectifier = envelope_detector(s_am, fs, fc)
            env_hilbert = hilbert_envelope_detector(s_am)
            env_peak = peak_envelope_detector(s_am, fs, fc)
            
            # Plot comparison
            fig_env = go.Figure()
            
            n_comp = min(int(0.05 * fs), len(envelope_detected))  # Show 50ms
            t_comp = np.arange(n_comp) / fs * 1000
            
            fig_env.add_trace(go.Scatter(
                x=t_comp, y=env_rectifier[:n_comp],
                mode='lines', name='Rectificador + Filtro',
                line=dict(color='green', width=2)
            ))
            
            fig_env.add_trace(go.Scatter(
                x=t_comp, y=env_hilbert[:n_comp],
                mode='lines', name='Transformada Hilbert',
                line=dict(color='blue', width=2)
            ))
            
            fig_env.add_trace(go.Scatter(
                x=t_comp, y=env_peak[:n_comp],
                mode='lines', name='Detector de Pico',
                line=dict(color='red', width=2)
            ))
            
            # Add theoretical envelope
            theoretical_env = 1 + mu * m[:n_comp]
            fig_env.add_trace(go.Scatter(
                x=t_comp, y=theoretical_env,
                mode='lines', name='Teórica',
                line=dict(color='black', width=3, dash='dash')
            ))
            
            fig_env.update_layout(
                title="Comparación de Métodos de Detección de Envolvente",
                xaxis_title="Tiempo (ms)",
                yaxis_title="Envolvente",
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_env, use_container_width=True)
            
            # Compute performance metrics for each method
            col1, col2, col3 = st.columns(3)
            
            with col1:
                snr_rect = compute_snr(theoretical_env, env_rectifier[:len(theoretical_env)])
                st.metric("📊 SNR Rectificador", f"{snr_rect:.1f} dB")
            
            with col2:
                snr_hilbert = compute_snr(theoretical_env, env_hilbert[:len(theoretical_env)])
                st.metric("📊 SNR Hilbert", f"{snr_hilbert:.1f} dB")
            
            with col3:
                snr_peak = compute_snr(theoretical_env, env_peak[:len(theoretical_env)])
                st.metric("📊 SNR Pico", f"{snr_peak:.1f} dB")
        
        # Envelope quality analysis
        st.markdown("**📈 Calidad de la envolvente:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Envelope smoothness (derivative analysis)
            env_diff = np.diff(envelope_detected)
            smoothness = np.std(env_diff)
            st.metric("📊 Suavidad envolvente", f"{smoothness:.6f}")
            
            # Envelope following accuracy
            theoretical_full = 1 + mu * m[:len(envelope_detected)]
            following_error = np.mean((envelope_detected - theoretical_full)**2)
            st.metric("❌ Error seguimiento", f"{following_error:.6f}")
        
        with col2:
            # Modulation index estimation accuracy
            mu_error = abs(mu_estimated - mu)
            st.metric("📐 Error estimación μ", f"{mu_error:.3f}")
            
            if mu_error < 0.05:
                st.success("✅ Estimación precisa")
            elif mu_error < 0.15:
                st.info("ℹ️ Estimación aceptable")
            else:
                st.warning("⚠️ Error significativo")
    
    with tab4:
        st.subheader("Análisis de distorsión por sobremodulación")
        
        if mu > 1.0:
            st.error(f"⚠️ **SOBREMODULACIÓN DETECTADA** (μ = {mu:.2f})")
            
            # Analyze envelope inversion points
            envelope_dc_removed = envelope_detected - np.mean(envelope_detected)
            inversion_points = np.where(envelope_dc_removed < 0)[0]
            
            if len(inversion_points) > 0:
                inversion_pct = len(inversion_points) / len(envelope_dc_removed) * 100
                st.metric("📊 % tiempo con inversión", f"{inversion_pct:.1f}%")
                
                # Show inversion locations
                fig_inversion = go.Figure()
                
                n_show = min(int(0.1 * fs), len(envelope_detected))
                t_show = np.arange(n_show) / fs * 1000
                
                fig_inversion.add_trace(go.Scatter(
                    x=t_show, y=envelope_detected[:n_show],
                    mode='lines', name='Envolvente detectada',
                    line=dict(color='red', width=2)
                ))
                
                # Mark zero line
                fig_inversion.add_hline(y=1.0, line_dash="dash", line_color="black",
                                      annotation_text="Nivel portadora")
                
                # Highlight inversion regions
                inversion_mask = envelope_detected[:n_show] < 1.0
                if np.any(inversion_mask):
                    inversion_times = t_show[inversion_mask]
                    inversion_values = envelope_detected[:n_show][inversion_mask]
                    
                    fig_inversion.add_trace(go.Scatter(
                        x=inversion_times, y=inversion_values,
                        mode='markers', name='Puntos de inversión',
                        marker=dict(color='yellow', size=8, symbol='x')
                    ))
                
                fig_inversion.update_layout(
                    title="Detección de Inversión de Envolvente",
                    xaxis_title="Tiempo (ms)",
                    yaxis_title="Envolvente",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_inversion, use_container_width=True)
            
            # Harmonic distortion analysis
            if signal_mode == "Triple tono (A + B + C)":
                st.markdown("**🎵 Análisis de distorsión armónica:**")
                
                # Compute THD for recovered signal
                fundamental_freqs = [f1, f2, f3]
                
                col1, col2, col3 = st.columns(3)
                
                for i, (freq, amp, col) in enumerate(zip(fundamental_freqs, [A, B, C], [col1, col2, col3])):
                    with col:
                        try:
                            sinad, snr_tone, thd = compute_sinad(m_recovered, fs, freq)
                            st.metric(f"THD {freq}Hz", f"{thd:.1f} dB")
                            
                            if thd > -40:
                                st.error("⚠️ Alta distorsión")
                            elif thd > -60:
                                st.warning("⚠️ Distorsión moderada")
                            else:
                                st.success("✅ Baja distorsión")
                        except:
                            st.text("No calculable")
        
        else:
            st.success("✅ **Sin sobremodulación** - Operación normal")
            st.info(f"Margen hasta sobremodulación: {((1.0 - mu) * 100):.1f}%")
            
            # Show margin to overmodulation
            fig_margin = go.Figure()
            
            mu_range = np.linspace(0.1, 1.5, 100)
            efficiency_curve = mu_range**2 / (2 + mu_range**2) * 100
            
            fig_margin.add_trace(go.Scatter(
                x=mu_range, y=efficiency_curve,
                mode='lines', name='Eficiencia AM',
                line=dict(color='blue', width=3)
            ))
            
            # Mark current operating point
            current_eff = mu**2 / (2 + mu**2) * 100
            fig_margin.add_trace(go.Scatter(
                x=[mu], y=[current_eff],
                mode='markers', name='Punto actual',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            # Mark overmodulation threshold
            fig_margin.add_vline(x=1.0, line_dash="dash", line_color="red",
                                annotation_text="Límite sobremodulación")
            
            fig_margin.update_layout(
                title="Eficiencia vs Índice de Modulación",
                xaxis_title="Índice de modulación (μ)",
                yaxis_title="Eficiencia (%)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_margin, use_container_width=True)
    
    with tab5:
        st.subheader("🎵 Audio AM - Original vs Recuperado")
        
        # Audio comparison
        create_audio_comparison_widget(
            m, m_recovered, fs,
            ("Mensaje Original", "Mensaje Recuperado")
        )
        
        # Audio quality analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Calidad de audio:**")
            
            # Perceptual quality metrics
            if len(m) == len(m_recovered):
                audio_snr = compute_snr(m, m_recovered)
                audio_corr = np.corrcoef(m, m_recovered)[0, 1]
                
                st.metric("🎵 SNR audio", f"{audio_snr:.1f} dB")
                st.metric("🔗 Correlación audio", f"{audio_corr:.4f}")
                
                # Perceptual quality estimate
                perceptual_quality = min(100, audio_corr * 100)
                st.metric("👂 Calidad perceptual", f"{perceptual_quality:.1f}%")
            
            # Dynamic range
            dr_original = compute_papr(m)
            dr_recovered = compute_papr(m_recovered)
            
            st.metric("📈 Rango dinámico orig.", f"{dr_original:.1f} dB")
            st.metric("📈 Rango dinámico rec.", f"{dr_recovered:.1f} dB")
        
        with col2:
            st.markdown("**🎛️ Efectos de la modulación:**")
            
            # Show effects of modulation parameters
            if mu > 1.0:
                st.error("⚠️ Sobremodulación causa distorsión audible")
                st.markdown("• Inversión de fase en la envolvente")
                st.markdown("• Armónicos espurios añadidos")
                st.markdown("• Pérdida de inteligibilidad")
            
            elif mu < 0.3:
                st.warning("⚠️ Baja modulación reduce calidad")
                st.markdown("• Pobre relación señal/ruido")
                st.markdown("• Aprovechamiento ineficiente de potencia")
            
            else:
                st.success("✅ Modulación óptima")
                st.markdown("• Buena calidad de audio")
                st.markdown("• Eficiencia aceptable")
                st.markdown("• Sin distorsión significativa")
        
        # AM signal audio (demonstration only)
        if st.checkbox("📻 Reproducir señal AM (solo demostración)"):
            st.warning("⚠️ La señal AM contiene componentes de RF no audibles")
            
            # Create audible AM by shifting to audio frequency
            fc_audio = 1000  # 1 kHz for audible demonstration
            s_am_audio = dsb_lc_modulate(m, mu, fc_audio, fs)
            
            st.markdown("**Señal AM en frecuencia audible (fc = 1 kHz):**")
            create_audio_player(s_am_audio, fs, "am_demo_audio", show_waveform=False)

# Helper functions for DSB-LC page

def peak_envelope_detector(signal: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Peak envelope detector implementation."""
    # Simple peak detector with exponential decay
    envelope = np.zeros_like(signal)
    decay_constant = np.exp(-2 * np.pi * fc / (10 * fs))  # Decay time constant
    
    for i in range(1, len(signal)):
        rectified = abs(signal[i])
        envelope[i] = max(rectified, envelope[i-1] * decay_constant)
    
    return envelope

def analyze_overmodulation_distortion(s_am: np.ndarray, mu: float, fc: float, fs: float) -> float:
    """Analyze distortion caused by overmodulation."""
    if mu <= 1.0:
        return 0.0
    
    # Detect envelope inversions
    envelope = hilbert_envelope_detector(s_am)
    envelope_normalized = (envelope - np.mean(envelope)) / np.std(envelope)
    
    # Count negative excursions (envelope inversions)
    inversions = np.sum(envelope_normalized < -2.0)  # 2-sigma threshold
    inversion_rate = inversions / len(envelope) * 100
    
    return inversion_rate

def estimate_am_bandwidth(signal: np.ndarray, fs: float) -> float:
    """Estimate bandwidth of message signal."""
    # Compute PSD and find bandwidth containing 99% of power
    f, psd_linear = compute_psd(signal, fs, 'welch')
    psd_linear = 10**(f[1:] / 10)  # Convert from dB to linear
    
    # Find frequency containing 99% of power
    total_power = np.sum(psd_linear)
    cumulative_power = np.cumsum(psd_linear)
    
    # Find 99% point
    idx_99 = np.where(cumulative_power >= 0.99 * total_power)[0]
    if len(idx_99) > 0:
        return f[idx_99[0] + 1]  # +1 to account for DC removal
    else:
        return fs / 4  # Fallback: quarter of sampling rate

if __name__ == "__main__":
    main()