"""
Fourier Series Analysis Page

Interactive analysis of textbook examples 3.6.1 through 3.6.4
with real-time parameter adjustment and visualization.

🔧 VERSIÓN CORREGIDA: Incluye selector de tipo de espectro
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dsp.fourier import *
from components.plotting import *
from components.layout import *
from utils.cache import *

st.set_page_config(page_title="Series de Fourier", page_icon="🌊", layout="wide")

def main():
    st.title("🌊 Series de Fourier - Ejemplos 3.6.1 a 3.6.4")
    
    st.markdown("""
    **Objetivo:** Analizar la reconstrucción de señales periódicas mediante series de Fourier
    y estudiar la convergencia de la serie según el número de armónicos.
    """)
    
    # Sidebar controls
    create_sidebar_header("Parámetros de Análisis")
    
    # Signal selection
    examples = {
        "3.6.1 - Triangular": "3.6.1",
        "3.6.2 - Diente de sierra": "3.6.2", 
        "3.6.3 - Parabólica": "3.6.3",
        "3.6.4 - Mixta (rampa + escalón)": "3.6.4"
    }
    
    selected_example = st.sidebar.selectbox(
        "🔍 Seleccionar señal:",
        list(examples.keys()),
        help="Ejemplos del libro de texto Introducción a Señales y Sistemas"
    )
    example_id = examples[selected_example]
    
    # Number of harmonics
    N = st.sidebar.slider(
        "🎵 Número de armónicos (N):",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Más armónicos = mejor aproximación pero más cálculo"
    )
    
    # Display parameters
    create_parameter_section("Visualización")
    
    periods = st.sidebar.slider(
        "📏 Períodos a mostrar:",
        min_value=1,
        max_value=5,
        value=2,
        help="Número de períodos en la gráfica temporal"
    )
    
    time_resolution = st.sidebar.selectbox(
        "🎯 Resolución temporal:",
        ["Normal (4K)", "Alta (8K)", "Ultra (16K)"],
        index=0,
        help="Mayor resolución = gráficas más suaves"
    )
    
    # Analysis options
    create_parameter_section("Análisis Avanzado")
    
    show_error = st.sidebar.checkbox(
        "📈 Análisis de convergencia", 
        value=False,
        help="Mostrar error RMS vs número de armónicos"
    )
    
    show_spectrum = st.sidebar.checkbox(
        "📊 Espectro de líneas", 
        value=True,
        help="Mostrar espectro de coeficientes de Fourier"
    )
    
    # 🔧 NUEVO: Selector de tipo de espectro
    spectrum_type = st.sidebar.selectbox(
        "📊 Tipo de espectro:",
        ["Trigonométrico (aₖ vs k)", "Complejo (|Cₖ| vs f)"],
        index=0,
        help="""
        • Trigonométrico: (aₖ, bₖ vs armónico k)
        • Complejo: Espectro bilateral (|Cₖ| vs frecuencia Hz)
        """
    )
    
    show_coefficients = st.sidebar.checkbox(
        "🔢 Tabla de coeficientes",
        value=False,
        help="Mostrar valores numéricos de los coeficientes"
    )
    
    # Set resolution based on selection
    resolution_map = {"Normal (4K)": 4096, "Alta (8K)": 8192, "Ultra (16K)": 16384}
    points_per_period = resolution_map[time_resolution]
    
    # Get signal parameters
    signal_params = get_signal_parameters(example_id)
    T = signal_params['period']
    omega0 = 2*np.pi/T
    
    # Create cache keys for expensive computations
    coeffs_key = create_cache_key(example_id, N)
    
    # Compute Fourier coefficients (cached)
    with st.spinner("Calculando coeficientes de Fourier..."):
        coeffs = cached_fourier_coefficients(example_id, N, coeffs_key)
    
    # 🔧 NUEVO: Agregar example_id a coeffs para plotting
    coeffs['example_id'] = example_id
    
    # Generate time vectors
    t_period = np.linspace(-T/2, T/2, points_per_period, endpoint=False)
    t_extended = np.linspace(-periods*T/2, periods*T/2, 
                           points_per_period*periods, endpoint=False)
    
    # Evaluate original signals
    x_original_period = eval_3_6_signal(example_id, t_period)
    x_original_extended = eval_3_6_signal_periodic(example_id, t_extended, T)
    
    # Reconstruct signal (cached)
    t_hash = hash_array(t_extended)
    coeffs_hash = hash_params(**coeffs)
    
    x_reconstructed = cached_signal_reconstruction(
        coeffs_hash, t_hash, T, _coeffs=coeffs, _t=t_extended
    )
    
    # Compute RMS error
    rms_error = compute_rms_error(x_original_extended, x_reconstructed)
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Período (T)", f"{T:.2f}")
    with col2:
        st.metric("🎵 Freq. fundamental", f"{1/T:.3f} Hz")
    with col3:
        st.metric("🔢 Armónicos", N)
    with col4:
        st.metric("❌ Error RMS", f"{rms_error:.6f}")
    
    # 🔧 NUEVO: Mostrar información específica del tipo de espectro
    if spectrum_type.startswith("Trigonométrico"):
        st.info("📊 **Modo Trigonométrico**: Mostrando coeficientes aₖ (y bₖ) vs armónico k")
    else:
        st.info("📊 **Modo Complejo**: Mostrando magnitud |Cₖ| vs frecuencia (Hz) - Espectro bilateral")
    
    # Main visualization
    if show_spectrum:
        col1, col2 = st.columns([1, 1])
    else:
        col1, col2 = st.columns([1, 0.01])
        col2 = None
    
    with col1:
        st.subheader("🔄 Reconstrucción temporal")
        
        # Create time comparison plot
        fig_time = create_time_comparison_plot(
            t_extended, x_original_extended, x_reconstructed, 
            f"Reconstrucción con N={N} armónicos"
        )
        
        # Add error fill
        error = x_original_extended - x_reconstructed
        fig_time.add_trace(go.Scatter(
            x=t_extended, y=error,
            mode='lines',
            name='Error',
            line=dict(color='red', width=1, dash='dot'),
            opacity=0.6,
            yaxis='y2',
            hovertemplate='Error: %{y:.4f}<extra></extra>'
        ))
        
        # Add secondary y-axis for error
        fig_time.update_layout(
            yaxis2=dict(
                title="Error",
                overlaying="y",
                side="right",
                range=[np.min(error)*1.1, np.max(error)*1.1]
            )
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Show convergence info based on symmetry
        if signal_params['symmetry'] == 'even':
            st.info("ℹ️ **Función par**: Solo coeficientes coseno (aₖ), bₖ = 0")
        elif signal_params['symmetry'] == 'odd':
            st.info("ℹ️ **Función impar**: Solo coeficientes seno (bₖ), aₖ = a₀ = 0")
        else:
            st.info("ℹ️ **Sin simetría**: Coeficientes tanto aₖ como bₖ presentes")
    
    if col2 is not None:
        with col2:
            st.subheader("📊 Espectro de líneas")
            
            # 🔧 NUEVO: Seleccionar tipo de espectro
            if spectrum_type.startswith("Trigonométrico"):
                spectrum_mode = "trigonometric"
            else:
                spectrum_mode = "complex"
            
            # Crear espectro según el tipo seleccionado
            fig_spectrum = create_line_spectrum_plot(coeffs, T, spectrum_mode)
            st.plotly_chart(fig_spectrum, use_container_width=True)
            
            # 🔧 NUEVO: Información contextual según el tipo de espectro
            if spectrum_mode == "trigonometric":
                st.markdown("**🎯 Coeficientes principales:**")
                
                # Mostrar coeficientes más importantes
                a_coeffs = coeffs['a_coeffs']
                b_coeffs = coeffs['b_coeffs']
                
                # Para funciones pares (como 3.6.1), mostrar aₖ
                if signal_params['symmetry'] == 'even':
                    for k in range(min(6, len(a_coeffs))):
                        if abs(a_coeffs[k]) > 1e-6:
                            coeff_name = "a₀" if k == 0 else f"a{k}"
                            st.text(f"{coeff_name} = {a_coeffs[k]:.6f}")
                
                # Para funciones impares, mostrar bₖ
                elif signal_params['symmetry'] == 'odd':
                    for k in range(1, min(6, len(b_coeffs))):
                        if abs(b_coeffs[k]) > 1e-6:
                            st.text(f"b{k} = {b_coeffs[k]:.6f}")
                
                # Para funciones generales, mostrar ambos
                else:
                    st.text("**Coeficientes aₖ (coseno):**")
                    for k in range(min(4, len(a_coeffs))):
                        if abs(a_coeffs[k]) > 1e-6:
                            coeff_name = "a₀" if k == 0 else f"a{k}"
                            st.text(f"  {coeff_name} = {a_coeffs[k]:.6f}")
                    
                    st.text("**Coeficientes bₖ (seno):**")
                    for k in range(1, min(4, len(b_coeffs))):
                        if abs(b_coeffs[k]) > 1e-6:
                            st.text(f"  b{k} = {b_coeffs[k]:.6f}")
            
            else:  # Modo complejo
                # Show dominant frequencies
                c_coeffs = coeffs['c_coeffs']
                magnitudes = np.abs(c_coeffs)
                N_coeffs = coeffs['N']
                
                # Find top 5 components (excluding DC)
                freqs = np.arange(-N_coeffs, N_coeffs+1) * omega0 / (2*np.pi)
                dc_idx = N_coeffs  # DC component index
                
                # Exclude DC for finding peaks
                mag_no_dc = magnitudes.copy()
                mag_no_dc[dc_idx] = 0
                
                top_indices = np.argsort(mag_no_dc)[-5:][::-1]
                
                st.markdown("**🎯 Componentes dominantes:**")
                # Mostrar DC primero
                if magnitudes[dc_idx] > 1e-6:
                    st.text(f"DC (f=0): |C₀| = {magnitudes[dc_idx]:.6f}")
                
                # Mostrar otros componentes
                for idx in top_indices:
                    if magnitudes[idx] > 1e-6:
                        freq = freqs[idx]
                        mag = magnitudes[idx]
                        k = idx - N_coeffs
                        st.text(f"f={freq:.3f} Hz (k={k}): |C{k}| = {mag:.6f}")
    
    # Convergence analysis
    if show_error:
        st.subheader("📈 Análisis de convergencia")
        
        # Use cached convergence analysis
        N_values, errors = cached_convergence_analysis(example_id, min(N+10, 50))
        
        fig_error = create_error_convergence_plot(N_values, errors)
        
        # Add theoretical convergence rate
        if example_id in ["3.6.1", "3.6.3"]:  # Continuous signals
            # Theoretical 1/N² convergence for smooth signals
            theoretical = errors[0] / N_values**2
            fig_error.add_trace(go.Scatter(
                x=N_values, y=theoretical,
                mode='lines',
                name='Teórico (1/N²)',
                line=dict(color='gray', dash='dash'),
                hovertemplate='N: %{x}<br>Teórico: %{y:.6f}<extra></extra>'
            ))
        
        st.plotly_chart(fig_error, use_container_width=True)
        
        # Convergence information
        col1, col2 = st.columns(2)
        with col1:
            if len(errors) > 1:
                improvement = (errors[0] - errors[-1]) / errors[0] * 100
                st.metric("📉 Mejora total", f"{improvement:.1f}%")
        
        with col2:
            if len(errors) > 10:
                recent_improvement = (errors[-10] - errors[-1]) / errors[-10] * 100
                st.metric("📉 Mejora reciente", f"{recent_improvement:.2f}%")
        
        # Gibbs phenomenon warning
        if example_id in ["3.6.2", "3.6.4"]:
            st.warning("⚠️ **Fenómeno de Gibbs**: Esta señal presenta discontinuidades. "
                      "El error cerca de las discontinuidades no disminuye por completo.")
    
    # Coefficients table
    if show_coefficients:
        st.subheader("🔢 Coeficientes de Fourier")
        
        # Create tabs for different coefficient representations
        tab1, tab2, tab3 = st.tabs(["Trigonométrica", "Exponencial", "Comparación"])
        
        with tab1:
            st.markdown("**Serie trigonométrica**: x(t) = a₀/2 + Σ[aₖcos(kω₀t) + bₖsin(kω₀t)]")
            
            # Create coefficients table
            coeff_data = []
            for k in range(min(N+1, 20)):  # Limit display to first 20
                ak = coeffs['a_coeffs'][k] if k < len(coeffs['a_coeffs']) else 0
                bk = coeffs['b_coeffs'][k] if k < len(coeffs['b_coeffs']) else 0
                
                coeff_data.append({
                    'k': k,
                    'aₖ': f"{ak:.6f}",
                    'bₖ': f"{bk:.6f}",
                    '|aₖ|': f"{abs(ak):.6f}",
                    '|bₖ|': f"{abs(bk):.6f}"
                })
            
            st.dataframe(coeff_data, use_container_width=True)
        
        with tab2:
            st.markdown("**Serie exponencial**: x(t) = ΣCₖe^(jkω₀t)")
            
            # Exponential coefficients
            exp_data = []
            c_coeffs = coeffs['c_coeffs']
            N_coeffs = coeffs['N']
            
            for i in range(len(c_coeffs)):
                k = i - N_coeffs  # Convert index to harmonic number
                ck = c_coeffs[i]
                
                exp_data.append({
                    'k': k,
                    'Cₖ (real)': f"{ck.real:.6f}",
                    'Cₖ (imag)': f"{ck.imag:.6f}",
                    '|Cₖ|': f"{abs(ck):.6f}",
                    '∠Cₖ (°)': f"{np.angle(ck, deg=True):.2f}"
                })
            
            st.dataframe(exp_data, use_container_width=True)
        
        with tab3:
            # Compare with analytical coefficients if available
            if signal_params.get('analytical_coeffs') is not None:
                st.markdown("**Comparación: Numérico vs Analítico**")
                
                analytical_func = signal_params['analytical_coeffs']
                comparison_data = []
                
                for k in range(1, min(N+1, 10)):
                    if signal_params['symmetry'] == 'even':
                        # Only a_k coefficients
                        numerical = coeffs['a_coeffs'][k]
                        analytical = analytical_func(k)
                        error = abs(numerical - analytical)
                        
                        comparison_data.append({
                            'k': k,
                            'aₖ (numérico)': f"{numerical:.6f}",
                            'aₖ (analítico)': f"{analytical:.6f}",
                            'Error': f"{error:.8f}",
                            'Error %': f"{error/abs(analytical)*100:.4f}" if analytical != 0 else "N/A"
                        })
                    elif signal_params['symmetry'] == 'odd':
                        # Only b_k coefficients
                        numerical = coeffs['b_coeffs'][k]
                        analytical = analytical_func(k)
                        error = abs(numerical - analytical)
                        
                        comparison_data.append({
                            'k': k,
                            'bₖ (numérico)': f"{numerical:.6f}",
                            'bₖ (analítico)': f"{analytical:.6f}",
                            'Error': f"{error:.8f}",
                            'Error %': f"{error/abs(analytical)*100:.4f}" if analytical != 0 else "N/A"
                        })
                
                if comparison_data:
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Calculate average error
                    errors = [float(row['Error']) for row in comparison_data]
                    avg_error = np.mean(errors)
                    st.metric("📊 Error promedio", f"{avg_error:.8f}")
            else:
                st.info("ℹ️ No hay fórmula analítica disponible para comparación")
    
    # Theoretical background
    with st.expander("📚 Fundamentos teóricos"):
        display_theoretical_background(example_id)
        
        # Add mathematical formulas
        st.markdown("---")
        st.markdown("### 📐 Fórmulas principales")
        
        st.markdown("""
        **Coeficientes trigonométricos:**
        - a₀ = (2/T)∫₀ᵀ x(t)dt
        - aₖ = (2/T)∫₀ᵀ x(t)cos(kω₀t)dt
        - bₖ = (2/T)∫₀ᵀ x(t)sin(kω₀t)dt
        
        **Coeficientes exponenciales:**
        - Cₖ = (1/T)∫₀ᵀ x(t)e^(-jkω₀t)dt
        - C₀ = a₀/2, Cₖ = (aₖ - jbₖ)/2, C₋ₖ = (aₖ + jbₖ)/2
        
        **Condiciones de convergencia (Dirichlet):**
        1. x(t) tiene un número finito de discontinuidades
        2. x(t) tiene un número finito de máximos y mínimos
        3. ∫|x(t)|dt < ∞
        """)
    
    # 🔧 NUEVO: Sección de debugging y verificación
    with st.expander("🔧 Verificación y debugging"):
        st.markdown("**🧪 Verificación de coeficientes para debugging:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Valores calculados:**")
            if len(coeffs['a_coeffs']) > 0:
                st.text(f"a₀ = {coeffs['a_coeffs'][0]:.6f}")
            if len(coeffs['a_coeffs']) > 1:
                st.text(f"a₁ = {coeffs['a_coeffs'][1]:.6f}")
            if len(coeffs['a_coeffs']) > 3:
                st.text(f"a₃ = {coeffs['a_coeffs'][3]:.6f}")
            
            if len(coeffs['c_coeffs']) > 0:
                N_c = coeffs['N']
                st.text(f"|C₀| = {abs(coeffs['c_coeffs'][N_c]):.6f}")
                if len(coeffs['c_coeffs']) > N_c + 1:
                    st.text(f"|C₁| = {abs(coeffs['c_coeffs'][N_c + 1]):.6f}")
        
        with col2:
            st.markdown("**Valores esperados (Ej. 3.6.1):**")
            st.text("a₀ = 1.000000")
            st.text("a₁ = 0.810569")
            st.text("a₃ = 0.090063")
            st.text("|C₀| = 0.500000")
            st.text("|C₁| = 0.405285")
        
        # Botón para debugging detallado
        if st.button("🔍 Ejecutar debugging detallado"):
            with st.spinner("Ejecutando verificación..."):
                # Aquí podrías llamar a la función de debugging
                st.code("""
# Para debugging detallado, ejecuta en tu entorno:
from dsp.fourier import debug_ejemplo_361
resultados = debug_ejemplo_361()
                """)
    
    # Performance information
    with st.expander("⚡ Información de rendimiento"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Parámetros actuales:**")
            st.text(f"Resolución: {points_per_period:,} puntos/período")
            st.text(f"Períodos: {periods}")
            st.text(f"Armónicos: {N}")
            st.text(f"Puntos totales: {len(t_extended):,}")
        
        with col2:
            st.markdown("**Estimación de memoria:**")
            memory_mb = estimate_array_memory(t_extended) + estimate_array_memory(x_original_extended) + estimate_array_memory(x_reconstructed)
            st.text(f"Memoria usada: {memory_mb:.1f} MB")
            st.text(f"Señal: {signal_params['description']}")
            st.text(f"Espectro: {spectrum_type}")
    
    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Exportar señales"):
            export_data = {
                'time': t_extended,
                'original': x_original_extended,
                'reconstructed': x_reconstructed,
                'error': x_original_extended - x_reconstructed,
                'sample_rate': 1/(t_extended[1] - t_extended[0])
            }
            
            # Save as NPZ
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                np.savez(tmp.name, **export_data)
                
                with open(tmp.name, 'rb') as f:
                    st.download_button(
                        label="📥 Descargar NPZ",
                        data=f.read(),
                        file_name=f"fourier_series_{example_id}_{N}harmonics.npz",
                        mime="application/octet-stream"
                    )
    
    with col2:
        if st.button("📊 Exportar coeficientes"):
            # Create CSV with coefficients
            import pandas as pd
            
            df_coeffs = pd.DataFrame({
                'k': range(len(coeffs['a_coeffs'])),
                'a_k': coeffs['a_coeffs'],
                'b_k': coeffs['b_coeffs']
            })
            
            csv = df_coeffs.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"fourier_coefficients_{example_id}_{N}harmonics.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🖼️ Exportar gráficas"):
            st.info("💡 Usa el botón de descarga en las gráficas interactivas (esquina superior derecha)")

if __name__ == "__main__":
    main()