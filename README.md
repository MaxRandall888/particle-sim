# Barnes–Hut 2-D Galaxy Simulator

A performant **Barnes–Hut** *N-body* simulator written in pure Python and accelerated with **Numba**. It offers real-time visualisation through **PyQtGraph** and MP4 generation.

> **Acknowledgement** – Core ideas and several implementation details were adapted from the open-source work of **[@alessialin](https://github.com/alessialin)** in the project *BarnesHut-py*.

---

## Features

* **Flat array quadtree** with breadth-first storage for cache efficiency
* **Single-precision (float32)** physics with the option of float64 centres of mass
* **Real-time rendering**: scatter or log-density render modes with interactive controls
* **Video export**: record trajectories directly to MP4 via Matplotlib/FFmpeg
* Comprehensive **pytest** suite for regression and numerical accuracy
* Cross-platform support for **Python 3.9 – 3.12** (Windows, macOS, Linux)

---

## Quick Start

```bash
git clone git@github.com:MaxRandall888/particle-sim.git
cd BarnesHut-sim

# (optional) create an isolated environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python barnes_hut.py           # launches the window
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:MaxRandall888/particle-sim.git
   cd BarnesHut-sim
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file pins **PyQt6** as the Qt backend. If preferred, replace the line with `PyQt5>=5.15` or `PySide6>=6.7`.

---

## Running the Simulation

**Simulation units:**
- **Length**: kiloparsecs (kpc)  
- **Mass**: thousands of solar masses (kMs, where 1 kMs = 10³ M☉)  
- **Time**: 10 Myr  

The gravitational constant **G** is provided by `body.gravity()` (defined in **body.py**) and has the value  
```python
half_box  = 100.0      # half-width of the periodic square (kpc)
theta     = 0.7        # Barnes–Hut opening angle (dimensionless)
epsilon   = 0.05       # Plummer softening length (kpc)
dt        = 0.05       # time step (10 Myr)
steps     = 5000       # total integration steps
plot_type = "density"  # "scatter" or "density"
```

Execute:

```bash
python barnes_hut.py
```

### Interactive Controls

| Key              | Action            |
| ---------------- | ----------------- |
| **Click & Drag** | Pan               |
| **Scroll**       | Zoom in / out     |

Any changes to the window (pan / zoom) will be displayed in the saved mp4.

---

## Generating Animations

The helper `utils.run_sim` enables exports:

```python
from utils import generate_galaxy, run_sim

bodies = generate_galaxy(
    r0=12.5, total_mass=50.0,
    n_bodies=20_000, half_box=100.0
)

run_sim(
    steps=4000,
    bodies=bodies,
    half_box=100.0,
    theta=0.5,
    epsilon=0.03,
    dt=0.04,
    point_size=2,
    plot_type="density",
    outfile="output/merger.mp4",
    fps=60
)
```

The directory `output/` is created automatically if it does not exist.

---

## Running the Test Suite

```bash
pytest -q        # terse
pytest -vv       # verbose
```

All graphical calls are mocked, so the tests run on headless CI servers without a display.

---

## Project Structure

```
.
├── array_quadtree.py       # Numba-JIT flat quadtree & force kernel
├── body.py                 # Lightweight particle class
├── barnes_hut.py           # Real-time simulation driver
├── utils.py                # IC generators, integrator, video export
├── test_barnes_hut_sim.py  # Pytest suite
├── requirements.txt
└── README.md
```

---

## Performance Guidance

* Enable persistent Numba compilation with `export NUMBA_CACHE=1` (Unix) or `set NUMBA_CACHE=1` (Windows).
* For very large systems (≥ 10⁵ bodies), consider enabling double-precision centres of mass (already supported in `array_quadtree.py`).
* The codebase is structured to allow straightforward porting of key kernels to **CuPy**, **PyTorch**, or other GPU frameworks for further acceleration.

---

## Known Issues

* There is a minor bug in the force calculation that leads to divergence from the realistic trajectory (no current fix).

---

## Contributing

Contributions are welcome. Please open an issue to propose major architectural or interface changes to ensure project continuity and scope alignment.

---

## License

This software is distributed under the **MIT License**.
Portions of the codebase are derived from *BarnesHut-py* by **Alessia Lin**, licensed under the MIT license.

---
