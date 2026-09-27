# Interpretable EEM framework for humic-acid reducing capacity

Code accompanying **“Fluorescence Encodes Reducing Capacity: The inferred electronic-structure changes of humic acid from Interpretable Deep Learning.”**

The workflow predicts humic-acid reducing capacity from excitation-emission matrices (EEMs) and supports model interpretation through Grad-RAM, manifold discovery and analysis (MDA), hierarchical clustering, and generalized 2D correlation spectroscopy (2D-COS).

## Installation

Two environments are used because MDA requires a different Python and TensorFlow stack.

Main environment:

```bash
conda env create -f RCPM_PREDICT.yml
conda activate RCPM_PREDICT
pip install -e . --no-deps
```

MDA environment:

```bash
conda env create -f MDA.yml
conda activate MDA
pip install -e . --no-deps
```

## Input data

Each EEM workbook contains emission wavelengths in the first row, excitation wavelengths in the first column, and fluorescence intensities in the remaining cells. The label workbook contains two columns:

- `filename`: EEM workbook name, including its extension;
- `label`: measured reducing capacity in meq L⁻¹.

Filenames begin with `PPHA`, `ESHA`, or `LHA` to identify the humic-acid source used for leave-one-source-out validation.

## Run the analysis

### 1 Prepare the EEM dataset

```bash
conda activate RCPM_PREDICT
python scripts/01_prepare_dataset.py --eem-dir data/raw/eem --labels data/raw/labels.xlsx --output-dir data/processed
```

### 2 Evaluate the prediction models

```bash
python scripts/02_evaluate.py --data-dir data/processed --output-dir outputs/evaluation --models plsr rf xgboost zero_cnn mask_aware_cnn masked_conv_cnn
```

### 3 Fit and interpret the RCPM

```bash
python scripts/03_interpret.py --data-dir data/processed --output-dir outputs/interpretation
```

### 4 Run MDA and clustering

```bash
conda activate MDA
python scripts/04_cluster.py --data-dir data/processed --interpret-dir outputs/interpretation --output-dir outputs/clustering
```

### 5 Run 2D-COS

```bash
conda activate RCPM_PREDICT
python scripts/05_twodcos.py --data-dir data/processed --clusters outputs/clustering/clusters_k3.csv --output-dir outputs/twodcos
```

Predictive performance is reported from leave-one-source-out validation. The model fitted to all samples is used only for Grad-RAM, MDA, and clustering.

## Documentation

- [Manuscript to code map](docs/MANUSCRIPT_CODE_MAP.md)
- [Release notes](docs/RELEASE_NOTES.md)
- [Third-party notices](docs/THIRD_PARTY_NOTICES.md)

## License

Original code in this repository is released under the [MIT License](LICENSE). The separately installed MDA implementation retains its upstream license; see the [third-party notices](docs/THIRD_PARTY_NOTICES.md).
