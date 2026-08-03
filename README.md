# Modulación de Ondas Theta por la Locomoción en el Hipocampo (Dataset CRCNS hc-3)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-11%20passed-success.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trabajo Final de Procesamiento Digital de Señales y Neurociencia Computacional.  
Análisis electrofisiológico de registros multicanal de Potenciales de Campo Local (LFP) y actividad neuronal de picos (*single-unit spikes*) en el hipocampo de roedores durante navegación espacial.

---

## 📌 Pregunta Científica y Marco Teórico

### 1. El Ritmo Theta como Reloj de la Navegación Espacial
Durante la locomoción activa y la exploración del entorno, el hipocampo de los mamíferos exhibe una prominente oscilación cuasi-sinusoidal regular de gran amplitud en la banda **Theta (4–12 Hz, centrada en ~8 Hz)**. Esta oscilación actúa como un metrónomo biológico que sincroniza los potenciales de acción neuronales (*phase-locking*) y permite la codificación temporal de la posición.

### 2. Postulado de Vanderwolf (1969)
> *"Hippocampal electrical activity and voluntary movement in the rat."* (Vanderwolf, C.H., 1969)  
> El ritmo Theta emerge exclusivamente durante el movimiento voluntario y la locomoción del animal (caminar, correr, explorar), colapsando hacia actividad irregular asincrónica de bajo voltaje (*Large Irregular Activity - LIA*) durante la inmovilidad o el reposo.

---

## 🔬 Metodología de Procesamiento de Señales

### 1. Criterios de Clasificación de Estados Conductuales
Para discriminar entre locomoción activa y reposo, se implementan dos aproximaciones complementarias:
1. **Biomarcador Electrofisiológico (LFP):**
   - Filtrado digital pasa-banda Butterworth de fase cero (orden 3, $4-12\text{ Hz}$, `filtfilt`).
   - Envolvente instantánea de Hilbert: $A_\theta(t) = |\mathcal{H}\{x_\theta(t)\}|$.
   - Ratio espectral $\Theta/\Delta(t) = \frac{P_\theta(t)}{P_\delta(t)}$.
   - **Locomoción:** $A_\theta(t) > 300\,\mu\text{V}$, $\Theta/\Delta > 5$.
   - **Inmovilidad:** $A_\theta(t) < 50\,\mu\text{V}$, $\Theta/\Delta < 0.5$.
2. **Cinemática del Animal (Video Tracking):**
   - Derivación temporal de la posición $(x, y)$: $v(t) = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$.
   - Locomoción: $v(t) > 5\text{ cm/s}$; Inmovilidad: $v(t) < 2\text{ cm/s}$.

### 2. Phase-Locking de Spikes Neuronales (Mizuseki et al., 2009)
Demodulación de fase instantánea mediante la transformada de Hilbert:
$$\theta(t) = \operatorname{atan2}(\mathcal{H}\{x_\theta(t)\}, x_\theta(t)) \in (-\pi, \pi]$$
Cálculo del **Vector Strength** (Longitud del vector medio resultante $R$) y Test de Circularidad de Rayleigh:
$$\bar{R} = \frac{1}{N} \left| \sum_{k=1}^N e^{j \theta_k} \right| = \frac{1}{N} \sqrt{\left(\sum \cos \theta_k\right)^2 + \left(\sum \sin \theta_k\right)^2}$$
$$z = N \bar{R}^2, \quad p \approx e^{-z}$$

---

## 📊 Resultados Principales

| Parámetro | Locomoción Activa ($t \approx 466\text{ s}$) | Inmovilidad / Reposo ($t \approx 2173\text{ s}$) | Ratio / Contraste |
|---|---|---|---|
| **Forma de Onda LFP** | Cuasi-sinusoidal regular ($\sim 7.8\text{ Hz}$) | Asincrónica / Ondas lentas (LIA) | — |
| **Amplitud Pico a Pico** | $> 1,000\,\mu\text{V}$ | $< 200\,\mu\text{V}$ | **$>5\times$** |
| **Densidad de Potencia (PSD)** | $40,304\,\mu\text{V}^2/\text{Hz}$ | $1,228\,\mu\text{V}^2/\text{Hz}$ | **$32.8\times$** |
| **Phase-Locking Neuronal** | Fase preferida $\mu \approx 2.76\text{ rad}$ ($158^\circ$) | — | **$p < 10^{-15}$** (Rayleigh) |

---

## 💻 Estructura del Repositorio

```text
hc3-reproducible-portfolio/
├── configs/
│   └── ec013_423.yaml              # Configuración de metadatos y canales
├── reports/
│   ├── presentation.html           # Presentación interactiva LaTeX Beamer (HTML5/CSS3/MathJax)
│   └── ec013.423/figures/          # Figuras de alta resolución (220 DPI)
│       ├── vanderwolf_zoom_contrast.png
│       ├── vanderwolf_theta_modulation.png
│       ├── phase_locking_mizuseki.png
│       └── psd.png
├── scripts/
│   ├── generate_true_contrast.py   # Generación de figuras de alto contraste activo vs reposo
│   ├── replicate_mizuseki_2009_theta.py  # Pipeline completo de LFP + Spikes
│   └── find_continuous_immobility.py     # Algoritmo de detección de estados
├── src/
│   └── hc3/                        # Paquete modular Python
│       ├── io.py                   # Lectores vectorizados binarios de memmap (.eeg, .res, .clu)
│       ├── lfp.py                  # Filtrado Butterworth de fase cero y PSD Welch
│       ├── spikes.py               # Demodulación de fase y estadística circular (Rayleigh)
│       └── qc.py                   # Control de calidad y detección de artefactos
├── tests/                          # 11 tests unitarios automatizados
└── pyproject.toml                  # Metadatos del paquete y dependencias
```

---

## 🚀 Instalación y Reproducción

### 1. Clonar el repositorio y configurar el entorno
```bash
git clone <URL_DEL_REPO>
cd hc3-reproducible-portfolio
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Ejecutar los Tests Automatizados
```bash
pytest tests/ -v
```

### 3. Generar las Figuras del Trabajo
```bash
python scripts/generate_true_contrast.py
python scripts/replicate_mizuseki_2009_theta.py
```

### 4. Abrir la Presentación Académica
Abrir `reports/presentation.html` en cualquier navegador web moderno.
- Presiona `F` para modo pantalla completa.
- Presiona `T` para alternar entre tema oscuro (*Dark Mode*) y tema claro (*Beamer Light*).

---

## 📚 Referencias Bibliográficas

1. **Vanderwolf, C. H. (1969).** *Hippocampal electrical activity and voluntary movement in the rat.* Electroencephalography and Clinical Neurophysiology, 26(4), 407–418.
2. **Buzsáki, G. (2002).** *Theta oscillations in the hippocampus.* Neuron, 33(3), 325–340.
3. **Mizuseki, K., Sirota, A., Pastalkova, E., & Buzsáki, G. (2009).** *Theta oscillations provide temporal windows for local circuit computation in the entorhinal-hippocampal loop.* Neuron, 64(2), 267–280.
4. **O'Keefe, J., & Nadel, L. (1978).** *The Hippocampus as a Cognitive Map.* Oxford University Press.
