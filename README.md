# 🔬 Laboratorio de Señales y Sistemas

Una aplicación interactiva en Streamlit que implementa las prácticas fundamentales de procesamiento de señales y comunicaciones.

## 📋 Descripción

Esta aplicación reproduce completamente el **Laboratorio 3** del curso de Señales y Sistemas, incluyendo:

1. **Series de Fourier** - Análisis de ejemplos 3.6.1 a 3.6.4
2. **Modulación DSB-SC** - Doble banda lateral con portadora suprimida  
3. **Modulación I/Q** - Cuadratura de amplitud (QAM)
4. **Modulación DSB-LC** - AM con portadora grande

## 🚀 Características Principales

- ✅ **Interfaz interactiva** con parámetros ajustables en tiempo real
- ✅ **Visualización avanzada** con gráficas interactivas Plotly
- ✅ **Procesamiento de audio** con reproducción y descarga
- ✅ **Análisis espectral completo** con PSDs normalizadas
- ✅ **Métricas de calidad** (SNR, THD, EVM, etc.)
- ✅ **Caché inteligente** para cálculos optimizados
- ✅ **Exportación de resultados** en múltiples formatos

## 🏗️ Estructura del Proyecto

```
signals-lab/
├── app.py                          # Aplicación principal Streamlit
├── pages/                          # Páginas del laboratorio
│   ├── 1_Fourier_Series.py        # Series de Fourier
│   ├── 2_DSB_SC_AM.py             # Modulación DSB-SC
│   ├── 3_IQ_QAM.py                # Modulación I/Q
│   └── 4_DSB_LC_AM.py             # Modulación DSB-LC (AM)
├── dsp/                            # Núcleo de procesamiento DSP
│   ├── fourier.py                 # Series de Fourier
│   ├── modulation.py              # Modulación/demodulación
│   ├── filters.py                 # Filtros digitales
│   └── metrics.py                 # Métricas de calidad
├── components/                     # Componentes de interfaz
│   ├── plotting.py                # Gráficas interactivas
│   ├── audio_widgets.py           # Widgets de audio
│   └── layout.py                  # Helpers de diseño
├── utils/                          # Utilidades
│   ├── cache.py                   # Sistema de caché
│   └── file_io.py                 # Entrada/salida de archivos
├── tests/                          # Pruebas unitarias
├── requirements.txt                # Dependencias Python
├── README.md                       # Esta documentación
└── .streamlit/config.toml         # Configuración Streamlit
```

## 🛠️ Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- Al menos 4 GB de RAM
- Navegador web moderno

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd signals-lab
```

2. **Crear entorno virtual:**
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación:**
```bash
streamlit run app.py
```

5. **Abrir en el navegador:**
   - La aplicación se abrirá automáticamente en `http://localhost:8501`
   - Si no se abre, navega manualmente a esa dirección

## 📚 Guía de Uso

### 1️⃣ Series de Fourier

**Objetivo:** Analizar la reconstrucción de señales periódicas mediante series de Fourier.

**Características:**
- 4 ejemplos del libro texto (3.6.1 - 3.6.4)
- Análisis de convergencia interactivo
- Comparación numérica vs analítica
- Visualización del fenómeno de Gibbs

**Parámetros ajustables:**
- Número de armónicos (1-100)
- Resolución temporal
- Períodos a mostrar
- Análisis de error RMS

### 2️⃣ Modulación DSB-SC

**Objetivo:** Implementar modulación de doble banda lateral con portadora suprimida.

**Características:**
- Carga de audio o señales demo
- Modulación y demodulación coherente
- Análisis espectral completo
- Métricas de calidad (SNR, correlación)

**Parámetros ajustables:**
- Frecuencia portadora (30-50 kHz)
- Orden del filtro receptor
- Frecuencia de muestreo
- Tipo de ventana del filtro

### 3️⃣ Modulación I/Q (QAM)

**Objetivo:** Transmitir dos señales simultáneamente usando ortogonalidad de seno/coseno.

**Características:**
- Dos canales independientes (I y Q)
- Análisis de crosstalk
- Simulación de errores de fase
- Diagrama de constelación

**Parámetros ajustables:**
- Señales demo para canales I y Q
- Error de fase y frecuencia
- Parámetros del filtro
- Análisis de aislamiento

### 4️⃣ Modulación DSB-LC (AM)

**Objetivo:** Modulación AM con diferentes índices de modulación.

**Características:**
- Triple tono configurable
- Índices μ = 0.7, 1.0, 1.2
- Detección de envolvente
- Análisis de sobremodulación

**Parámetros ajustables:**
- Amplitudes y frecuencias A, B, C
- Índice de modulación μ
- Método de detección de envolvente
- Análisis de distorsión

## 🧮 Fundamentos Matemáticos

### Series de Fourier
```
x(t) = a₀/2 + Σ[aₖcos(kω₀t) + bₖsin(kω₀t)]

aₖ = (2/T)∫₀ᵀ x(t)cos(kω₀t)dt
bₖ = (2/T)∫₀ᵀ x(t)sin(kω₀t)dt
```

### Modulación DSB-SC
```
y(t) = x(t) × cos(ωct) ↔ ½[X(f-fc) + X(f+fc)]
Demodulación: x̂(t) = LPF{2y(t) × cos(ωct)}
```

### Modulación I/Q
```
s(t) = xI(t)cos(ωct) + xQ(t)sin(ωct)
Ortogonalidad: ∫cos(ωct)sin(ωct)dt = 0
```

### AM DSB-LC
```
s(t) = [1 + μm(t)] × cos(ωct)
μ = (Emax - Emin) / (Emax + Emin)
```

## 📊 Métricas Implementadas

- **SNR** (Signal-to-Noise Ratio)
- **THD** (Total Harmonic Distortion)
- **EVM** (Error Vector Magnitude)
- **SINAD** (Signal-to-Noise-and-Distortion)
- **Correlación cruzada**
- **Error RMS**
- **Eficiencia espectral**
- **PAPR** (Peak-to-Average Power Ratio)

## 🎛️ Controles Avanzados

### Caché Inteligente
- Optimización automática de cálculos repetitivos
- Gestión de memoria eficiente
- Invalidación automática cuando cambian parámetros

### Filtros Digitales
- FIR con ventanas (Hamming, Kaiser, Blackman)
- IIR Butterworth para receptores
- Filtros adaptativos para cancelación de ruido

### Procesamiento de Audio
- Soporte para WAV, MP3, FLAC, OGG
- Conversión automática estéreo→mono
- Remuestreo inteligente para RF
- Reproducción y descarga de resultados

## 🔧 Personalización

### Añadir Nuevas Señales Demo
```python
# En dsp/modulation.py, función generate_demo_signal()
elif signal_type == "Mi nueva señal":
    x = mi_funcion_generadora(t, parametros)
    return x, fs
```

### Crear Nuevos Filtros
```python
# En dsp/filters.py
def mi_filtro_personalizado(cutoff, fs, order):
    # Implementación del filtro
    return coeficientes
```

### Agregar Métricas
```python
# En dsp/metrics.py
def mi_metrica_personalizada(signal1, signal2):
    # Cálculo de la métrica
    return valor_metrica
```

## 🧪 Pruebas

Ejecutar las pruebas unitarias:

```bash
# Instalar pytest si no está incluido
pip install pytest

# Ejecutar todas las pruebas
pytest tests/

# Ejecutar pruebas específicas
pytest tests/test_fourier.py
pytest tests/test_modulation.py
```

### Cobertura de Pruebas
- ✅ Coeficientes de Fourier vs fórmulas analíticas
- ✅ SNR de recuperación DSB-SC > 60 dB
- ✅ Crosstalk I/Q < -20 dB con fase perfecta
- ✅ Detección correcta de sobremodulación
- ✅ Filtros dentro de especificaciones

## 🚀 Despliegue

### Streamlit Cloud
1. Subir código a GitHub
2. Conectar con Streamlit Cloud
3. La aplicación se despliega automáticamente

### Docker (Opcional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### Heroku
```bash
# Crear archivos necesarios
echo "web: streamlit run app.py --server.port=$PORT" > Procfile
echo "python-3.9.0" > runtime.txt

# Desplegar
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🐛 Solución de Problemas

### Problemas Comunes

**Error: "ModuleNotFoundError"**
```bash
# Verificar que el entorno virtual está activado
pip install -r requirements.txt
```

**La aplicación va lenta**
```python
# Reducir resolución temporal en la barra lateral
# Limpiar caché: Sidebar → Cache Management → Clear All Caches
```

**Audio no se reproduce**
```bash
# Verificar formato de audio soportado
# Convertir a WAV si es necesario
```

**Gráficas no se muestran**
```bash
# Verificar que Plotly está instalado correctamente
pip install --upgrade plotly
```

### Optimización de Rendimiento

1. **Usar caché eficientemente:**
   - Los parámetros se cachean automáticamente
   - Cambios menores no invalidan todo el caché

2. **Limitar duración de audio:**
   - Máximo recomendado: 15 segundos
   - Para archivos largos, la app cortará automáticamente

3. **Ajustar resolución:**
   - Normal (4K): Uso general
   - Alta (8K): Para análisis detallado
   - Ultra (16K): Solo para casos especiales

## 📖 Referencias

### Libros de Texto
- **Introducción a las Señales y Sistemas** - Ejemplos 3.6.1 a 3.6.4
- **Principles of Communications** - B.P. Lathi
- **Digital Signal Processing** - Proakis & Manolakis

### Documentación Técnica
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NumPy/SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Plotly Interactive Plots](https://plotly.com/python/)

### Estándares Relacionados
- IEEE 802.11 (WiFi) - Usa modulación I/Q
- AM Radio Broadcasting - DSB-LC implementation
- Digital TV Standards - Advanced modulation schemes

## 🤝 Contribución

### Cómo Contribuir
1. Fork el repositorio
2. Crear rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abrir Pull Request

### Estilo de Código
- Seguir PEP 8 para Python
- Usar docstrings estilo Google
- Comentarios en español para funciones principales
- Tests unitarios para funciones críticas

### Áreas de Mejora
- [ ] Modulación FM y PM
- [ ] Filtros adaptativos avanzados
- [ ] Análisis de BER para señales digitales
- [ ] Simulación de canal con ruido AWGN
- [ ] Estimación de parámetros ciegos
- [ ] Sincronización de portadora y reloj

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

## 👥 Autores

- **Desarrollo Principal** - Implementación completa del laboratorio - Sebastián Escobar

## 🙏 Agradecimientos

- Profesores del curso de Señales y Sistemas (Juan Pablo Tello)
- Comunidad de Streamlit por la excelente documentación
- Desarrolladores de SciPy/NumPy por las herramientas DSP

---

**¡Disfruta explorando el fascinante mundo del procesamiento de señales! 🎵📡**