# VAE Implementation with SCVI

A **custom Variational Autoencoder (VAE)** implementation for single-cell RNA-seq data, built on top of [scVI](https://scvi-tools.org/). This project compares a custom VAE adaptation with the base scVI model, focusing on gene expression reconstruction and latent space analysis.

---

## 📌 About

This repository contains:

- A **custom VAE model** (`modules/custom_module.py`) adapted from scVI’s tutorial.
- A **base scVI VAE model** (`modules/scvi_Modul.py`) for comparison.
- Notebooks for **training** (`notebooks/Training_two_models.ipynb`) and **comparing models** (`notebooks/SCVI_comparing_models.ipynb`) using UMAP visualizations.
- Pre-trained models (`base_model/`, `my_custom_model/`).

---

## 🛠️ Setup

### Prerequisites

- Python ≥ 3.8
- PyTorch
- scvi-tools
- scanpy
- anndata
- Jupyter Notebook

### Installation

1. Clone the repository:
  ```bash
   git clone https://github.com/captainPopec/VAE-implementation-with-SCVI.git
   cd VAE-implementation-with-SCVI
  ```
2. Create the conda environment:
  ```bash
   conda env create -f env.yml
   conda activate scvi-env
  ```
3. Install the package in development mode (optional, but recommended):
  ```bash
   pip install -e .
  ```

---

## 🚀 Usage

### 1. Train the Models

Open and run the training notebook:

```bash
jupyter notebook notebooks/Training_two_models.ipynb
```

This will train both the **base scVI model** and your **custom VAE model** on the provided dataset.

### 2. Compare Models

Use the comparison notebook to generate UMAP plots:

```bash
jupyter notebook notebooks/SCVI_comparing_models.ipynb
```

### 3. Load Pre-trained Models

Pre-trained models are available in:

- `base_model/`
- `my_custom_model/`

### 4. Use as a Python Package

If you installed the package in development mode, you can import modules directly:

```python
from modules.custom_module import CustomVAE  # Example: Replace CustomVAE with your class name
from modules.scvi_Modul import BaseSCVIModel  # Example: Replace BaseSCVIModel with your class name
```

---

## 📂 Repository Structure

```
VAE-implementation-with-SCVI/
├── modules/                  # Main Python modules
│   ├── __init__.py
│   ├── custom_module.py      # Custom VAE implementation
│   └── scvi_Modul.py         # Base scVI VAE (for comparison)
│
├── notebooks/                # Jupyter notebooks
│   ├── Training_two_models.ipynb   # Training script for both models
│   └── SCVI_comparing_models.ipynb # Model comparison and UMAP visualization
│
├── base_model/               # Pre-trained base scVI model
├── my_custom_model/          # Pre-trained custom VAE model
│
├── env.yml                   # Conda environment file
└── README.md                 # Project documentation
```

---

## 🔍 Key Features

- **Custom VAE Architecture**: Adapted from scVI’s tutorial, with direct integration.
- **Model Comparison**: Tools to compare the base scVI model with your custom implementation.
- **UMAP Visualization**: Notebooks to visualize latent spaces and reconstruction quality.
- **Modular Design**: Clean separation of models, notebooks, and data.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for:

- Bug fixes.
- New features (e.g., additional loss functions, architectures).
- Documentation improvements.

---

## 📜 License

This project is licensed under the MIT License.
