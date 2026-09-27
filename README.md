# Interpretable EEM framework for humic-acid reducing capacity

This repository contains the analysis workflow accompanying **“Fluorescence Encodes Reducing Capacity: The inferred electronic-structure changes of humic acid from Interpretable Deep Learning.”** It predicts solution reducing capacity from excitation-emission matrices (EEMs), evaluates transfer to a humic-acid source excluded from training, and links the learned representation to spectral reorganization.

## Workflow overview

The workflow includes:

- direct coordinate assignment to an 81 × 81 Ex/Em grid without interpolation;
- explicit preservation of unmeasured positions as `NaN` and a binary coverage mask;
- leave-one-source-out (LOSO) validation across PPHA, ESHA, and LHA;
- a common-input comparison of PLSR, random forest, XGBoost, and a zero-filled CNN;
- sensitivity analyses using mask-aware and masked-convolution CNNs;
- Grad-RAM, manifold discovery and analysis (MDA), Ward clustering, and generalized 2D-COS.

## Repository layout

```text
src/eem_rc/
  data.py          EEM alignment, mask construction, and dataset I/O
  models.py        CNN architectures and masked convolution
  evaluate.py      LOSO evaluation for baselines and CNN variants
  interpret.py     Full-data RCPM fit, Grad-RAM, and feature extraction
  clustering.py    MDA package call, Ward clustering, and epsilon-squared selection
  twodcos.py       Cluster-pair generalized 2D correlation spectroscopy
scripts/           Command-line entry scripts
tests/             Fast tests using synthetic data
data/              Local inputs; large/private files are not committed
outputs/           Generated models, tables, and figures
```

## Installation

The workflow uses two environments because the RCPM analysis and the MDA implementation require different TensorFlow and Python versions.

Create the main analysis environment:

```bash
conda env create -f RCPM_PREDICT.yml
conda activate RCPM_PREDICT
pip install -e . --no-deps
```

Create the separate environment used only for the MDA projection and its immediately coupled clustering step:

```bash
conda env create -f MDA.yml
conda activate MDA
pip install -e . --no-deps
```

`RCPM_PREDICT` reproduces the study's Python 3.9/TensorFlow 2.19 workflow. `MDA` reproduces the Python 3.10/TensorFlow 2.8.2 environment used by the upstream MDA implementation.

## Expected input

Each EEM workbook contains emission wavelengths in the first row, excitation wavelengths in the first column, and fluorescence intensity values in the remaining cells, matching the orientation used in the original study files. The label workbook must contain:

| column | meaning |
|---|---|
| `filename` | workbook filename, including extension |
| `label` | measured reducing capacity in meq L⁻¹ |

Filenames must begin with `PPHA`, `ESHA`, or `LHA` so that the LOSO groups can be reconstructed.

## Reproduce the workflow

1. Build the aligned dataset:

```bash
python scripts/01_prepare_dataset.py --eem-dir data/raw/eem --labels data/raw/labels.xlsx --output-dir data/processed
```

2. Run the common zero-filled benchmark:

```bash
python scripts/02_evaluate.py --data-dir data/processed --output-dir outputs/benchmark --models plsr rf xgboost zero_cnn
```

3. Run the missing-region sensitivity analysis:

```bash
python scripts/02_evaluate.py --data-dir data/processed --output-dir outputs/mask_sensitivity --models mask_aware_cnn masked_conv_cnn
```

4. Fit the full-data mask-aware RCPM and extract interpretation products:

```bash
python scripts/03_interpret.py --data-dir data/processed --output-dir outputs/interpretation
```

5. Switch to the dedicated MDA environment, then run MDA and hierarchical clustering:

```bash
conda activate MDA
python scripts/04_cluster.py --data-dir data/processed --interpret-dir outputs/interpretation --output-dir outputs/clustering
```

6. Return to the main environment and run 2D-COS for the C1-C2 and C2-C3 transitions at Em = 425 and 490 nm:

```bash
conda activate RCPM_PREDICT
python scripts/05_twodcos.py --data-dir data/processed --clusters outputs/clustering/clusters_k3.csv --output-dir outputs/twodcos
```

The full-data model is used only to create a common feature space for interpretation. Predictive performance must be reported from the LOSO results.

## Data availability

The study dataset is not publicly available at this stage. It may be made available by the corresponding author upon reasonable request, subject to applicable research and institutional requirements. Raw EEM workbooks, labels, trained model files, and generated analysis outputs are not included in this repository. Authorized users should place local input files under `data/raw/`.

## Reproducibility notes

- TensorFlow and NumPy seeds are fixed, but exact GPU results can vary across hardware and library builds.
- Feature scaling and label scaling are fitted only on each training subset during LOSO validation.
- The mask channel is never intensity-normalized; only the fluorescence channel is divided by the training maximum.
- Cluster labels are reordered as C1-C3 by increasing mean measured reducing capacity.

## Citation

Please cite the associated article. Bibliographic details and DOI will be added after publication.

## License

The source code is released under the [MIT License](LICENSE). Users may use, modify, and redistribute the code, including for commercial purposes, provided that the copyright and license notices are retained. Academic users are requested to cite the associated article when using this workflow in published research.

The separately installed MDA implementation is third-party software and is not covered by this repository's MIT License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its citation, source, and upstream license restrictions.
