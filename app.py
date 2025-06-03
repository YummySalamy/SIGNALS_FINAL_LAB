import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Laboratorio de Señales y Sistemas",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main > div {
        padding: 2rem 1rem;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .formula-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007acc;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔬 Laboratorio de Señales y Sistemas")
    st.markdown("---")
    
    st.markdown("""
    ## 📚 Contenido del Laboratorio
    
    Este laboratorio implementa las prácticas fundamentales de procesamiento de señales:
    
    ### 1️⃣ Series de Fourier
    - Ejemplos 3.6.1 - 3.6.4 del libro texto
    - Reconstrucción de señales periódicas
    - Análisis de convergencia y error RMS
    
    ### 2️⃣ Modulación DSB-SC 
    - Modulación de doble banda lateral con portadora suprimida
    - Demodulación coherente con filtro pasa-bajas
    - Análisis espectral y SNR
    
    ### 3️⃣ Modulación I/Q (QAM)
    - Multiplexación en cuadratura
    - Transmisión simultánea de dos señales
    - Demodulación ortogonal
    
    ### 4️⃣ Modulación DSB-LC (AM)
    - Modulación de amplitud con portadora grande
    - Diferentes índices de modulación (μ = 0.7, 1.0, 1.2)
    - Detección de envolvente
    
    ## 🚀 Instrucciones
    
    1. Selecciona una página del laboratorio desde la barra lateral
    2. Ajusta los parámetros según tus necesidades
    3. Observa los resultados en tiempo real
    4. Descarga las señales procesadas
    
    ## 📊 Fundamentos Teóricos
    """)
    
    # Theoretical formulas in styled boxes
    st.markdown("""
    <div class="formula-box">
    <strong>Serie de Fourier Trigonométrica:</strong><br>
    x(t) = a₀/2 + Σ[aₖcos(kω₀t) + bₖsin(kω₀t)]<br>
    donde: aₖ = (2/T)∫x(t)cos(kω₀t)dt, bₖ = (2/T)∫x(t)sin(kω₀t)dt
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="formula-box">
    <strong>Modulación DSB-SC:</strong><br>
    y(t) = x(t)cos(ωct) ↔ ½[X(f-fc) + X(f+fc)]<br>
    Demodulación coherente: x̂(t) = LPF{2y(t)cos(ωct)}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="formula-box">
    <strong>Modulación I/Q (QAM):</strong><br>
    s(t) = xᵢ(t)cos(ωct) + xQ(t)sin(ωct)<br>
    Ortogonalidad: ∫cos(ωct)sin(ωct)dt = 0
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="formula-box">
    <strong>AM DSB-LC:</strong><br>
    s(t) = [1 + μm(t)]cos(ωct)<br>
    μ > 1 produce sobremodulación y distorsión
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Para empezar rápidamente:
        
        1. **Series de Fourier**: Explora la reconstrucción de señales clásicas
        2. **DSB-SC**: Sube tu audio favorito y observa la modulación
        3. **I/Q**: Transmite dos señales al mismo tiempo
        4. **DSB-LC**: Experimenta con diferentes índices de modulación
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Características principales:
        
        - ✅ Parámetros ajustables en tiempo real
        - ✅ Visualización interactiva con Plotly
        - ✅ Reproducción y descarga de audio
        - ✅ Análisis espectral completo
        - ✅ Métricas de calidad (SNR, error RMS)
        """)
    
    st.info("📌 Navega por las páginas usando el menú lateral para comenzar con los experimentos.")
    
    # Additional information
    with st.expander("ℹ️ Información técnica"):
        st.markdown("""
        **Rangos de frecuencia recomendados:**
        - Frecuencia portadora: 30-50 kHz
        - Frecuencia de muestreo: 100-200 kHz (para RF)
        - Audio base: 48 kHz estándar
        
        **Formatos de audio soportados:**
        WAV, MP3, FLAC, OGG (se convierten automáticamente a mono)
        
        **Filtros implementados:**
        - FIR Hamming window para recepción
        - Órdenes configurables (51-1001 taps)
        - Frecuencias de corte adaptativas
        """)

if __name__ == "__main__":
    main()