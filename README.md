# Physics-Informed Neural Networks for Collective Cancer Invasion

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of physics-informed neural networks (PINNs) and Fourier neural operators (FNOs) for modeling collective cancer invasion dynamics. This project solves coupled nonlinear partial differential equations describing the spatiotemporal evolution of leader and follower cancer cell populations using advanced machine learning techniques.

## 🎯 Overview

This application models the complex dynamics of cancer cell invasion through a coupled system of reaction-diffusion equations. The mathematical model captures the nonlinear interactions between leader cells (invasive front) and follower cells (main tumor mass), solved using physics-informed learning approaches that enforce PDE constraints during neural network training.

### Key Features

- **🔬 Physics-Informed Learning**: Direct enforcement of PDE constraints in the loss function
- **🌊 Fourier Neural Operators**: Spectral-based architecture for efficient PDE solving
- **📊 Multi-Format Results**: Saves predictions in NumPy, CSV, and JSON formats
- **🎨 Advanced Visualization**: High-quality plots and time-evolution snapshots
- **⚙️ Configurable Architecture**: JSON-based configuration for all parameters
- **🏗️ Modular Design**: Clean separation of concerns with object-oriented architecture

## 📋 Table of Contents

- [Mathematical Model](#-mathematical-model)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Results Analysis](#-results-analysis)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)

## 🔬 Mathematical Model

The model describes the spatiotemporal evolution of leader ($\rho_l$) and follower ($\rho_f$) cancer cell densities through coupled reaction-diffusion equations:

### PDE System
```
∂ρ_l/∂t = D_l ∇²ρ_l - α_lf ρ_l ρ_f + K_l ρ_l (1 - ρ_l - ρ_f)
∂ρ_f/∂t = D_f ∇²ρ_f + α_lf ρ_l ρ_f + K_f ρ_f (1 - ρ_l - ρ_f)
```

### Boundary Conditions
- **Left boundary** (x=0): No-flux conditions
- **Right boundary** (x=1): Zero density conditions
- **Initial conditions**: Gaussian distributions for both species

### Parameters
- **D_l, D_f**: Diffusion coefficients for leader and follower cells
- **α_lf**: Coupling strength between species
- **K_l, K_f**: Growth rates
- **X**: Coupling parameter

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- PyTorch 2.9+ with CUDA support (recommended for GPU acceleration)
- NVIDIA GPU with CUDA 12.8+ (optional but recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/pinn-cancer-invasion.git
cd pinn-cancer-invasion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CUDA-enabled PyTorch (recommended)
pip install -r requirements-cuda.txt

# Install remaining dependencies
pip install -r requirements.txt
```

### Alternative Installation (CPU-only)
If you don't have a CUDA-compatible GPU:
```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 💻 Usage

### Basic Execution
```bash
# Run with default configuration
python main.py
```

### Custom Configuration
Edit `config.json` to modify parameters:
```json
{
  "model": {
    "type": "FNO",
    "fno": {
      "modes": 16,
      "width": 64,
      "n_layers": 4
    }
  },
  "training": {
    "n_epochs": 2500,
    "learning_rate": 0.0005
  }
}
```

### Output Structure
```
results/
├── model_weights.pth          # Trained model weights
├── predictions.npz            # Compressed numpy arrays
├── predictions.csv            # Tabular data format
└── results_summary.json       # Statistical summary

plots/
└── density_snapshot.png       # Final state visualization
```

## 📁 Project Structure

```
pinn-cancer-invasion/
├── main.py                     # Application entry point
├── controller.py               # Simulation orchestrator (Singleton)
├── config.json                 # Configuration parameters
├── pyproject.toml             # Project metadata
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── scripts/
│   ├── models.py              # Neural network architectures
│   ├── trainer.py             # Training logic and PDE computations
│   ├── data_loader.py         # Synthetic data generation
│   ├── visualization.py       # Plotting and result saving
│   ├── utils.py               # Device detection utilities
│   └── analyze_results.py     # Post-processing analysis
├── results/                   # Generated results (auto-created)
└── plots/                     # Generated visualizations (auto-created)
```

## ⚙️ Configuration

The `config.json` file controls all aspects of the simulation:

### Model Configuration
```json
{
  "model": {
    "type": "FNO",              // "PINN" or "FNO"
    "fno": {
      "modes": 16,             // Fourier modes
      "width": 64,             // Hidden dimension
      "n_layers": 4            // Network depth
    }
  }
}
```

### Training Parameters
```json
{
  "training": {
    "n_epochs": 2500,          // Training iterations
    "learning_rate": 0.0005,   // Optimizer learning rate
    "loss_weights": {          // Physics-informed loss weights
      "physics": 10.0,
      "ic": 7.0,
      "bc_left": 2.0,
      "bc_right": 1.0
    }
  }
}
```

### Physical Parameters
```json
{
  "constants": {
    "D_l": 0.01,               // Leader diffusion
    "D_f": 0.01,               // Follower diffusion
    "a_lf": 1.0,               // Species coupling
    "K_l": 1.0,                // Leader growth
    "K_f": 1.0,                // Follower growth
    "X": 0.2                   // Coupling parameter
  }
}
```

## 📊 Results Analysis

### Post-Processing Analysis
```bash
# Analyze saved results without retraining
python scripts/analyze_results.py
```

### Results Formats

**NumPy Arrays** (`predictions.npz`):
- Compressed binary format for numerical analysis
- Contains full spatiotemporal density fields
- Fast loading for further processing

**CSV Data** (`predictions.csv`):
- Human-readable tabular format
- Compatible with spreadsheet applications
- Includes spatial coordinates and time stamps

**JSON Summary** (`results_summary.json`):
- Statistical overview of simulation results
- Min/max/mean values for both species
- Metadata for reproducibility

**Visualizations** (`plots/`):
- High-resolution PNG snapshots
- Time-annotated density profiles
- Publication-ready figures

## 🏗️ Architecture

### Design Principles
- **Modular Structure**: Clear separation of concerns
- **Singleton Pattern**: Centralized simulation control
- **Configuration-Driven**: Externalized parameters
- **Error Resilience**: Comprehensive exception handling
- **Type Safety**: Full type hinting throughout

### Core Components

**SimulationController** (`controller.py`):
- Orchestrates the complete workflow
- Manages data preparation, training, and visualization
- Implements singleton pattern for state consistency

**Neural Architectures** (`models.py`):
- `CoupledSpeciesTFNO`: Fourier neural operator for PDE solving
- Modular design supporting different architectures

**Training Engine** (`trainer.py`):
- Physics-informed loss computation
- Automatic differentiation for PDE residuals
- Gradient-based optimization

**Data Pipeline** (`data_loader.py`):
- Synthetic data generation
- Initial and boundary condition enforcement
- PyTorch tensor preparation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research in physics-informed neural networks
- Implements methods from the Neural Operator literature
- Inspired by collective behavior in biological systems

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

2. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

---

**Note**: This implementation is for research and educational purposes. For medical applications, please consult with domain experts and validate against experimental data.