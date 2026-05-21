# VAE Implementation with SCVI

A **custom Variational Autoencoder (VAE)** implementation for single-cell RNA-seq data, built on top of [scVI](https://scvi-tools.org/). This project compares a custom VAE adaptation with the base scVI model, focusing on gene expression reconstruction and latent space analysis.

---

## 📌 About

This repository contains:

- A **custom VAE model** (`MojModul.py`) adapted from scVI’s tutorial.
- A **wrapped version** of both the custom and base scVI models (`MojWrappedmodel.py`, `scvi_wrapped.py`).
- Notebooks for **training** (`Training_two_models.ipynb`) and **comparing models** (`SCVI_comparing_models.ipynb`) using UMAP visualizations.
- Pre-trained models (`base_model/`, `my_custom_model/`).
- Example dataset (`kang_counts_25k.h5ad`).

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

---

## 🚀 Usage

### 1. Train the Models

Open and run the notebook:

```bash
jupyter notebook Training_two_models.ipynb
```

This will train both the **base scVI model** and your **custom VAE model** on the provided dataset.

### 2. Compare Models

Use the comparison notebook to generate UMAP plots:

```bash
jupyter notebook SCVI_comparing_models.ipynb
```

### 3. Load Pre-trained Models

Pre-trained models are available in:

- `base_model/`
- `my_custom_model/`

---



## 🔍 Key Features

- **Custom VAE Architecture**: Adapted from scVI’s tutorial, with wrappers for seamless integration.
- **Model Comparison**: Tools to compare the base scVI model with your custom implementation.
- **UMAP Visualization**: Notebooks to visualize latent spaces and reconstruction quality.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for:

- Bug fixes.
- New features (e.g., additional loss functions, architectures).
- Documentation improvements.

---

## 📜 License

This project is licensed under the MIT License.
