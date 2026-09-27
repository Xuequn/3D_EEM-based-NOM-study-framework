# Manuscript to code map

| Manuscript method | Implementation | Main output |
|---|---|---|
| EEM alignment to 200–600 nm at 5 nm intervals | `src/eem_rc/data.py` | `eem_features_raw.npy` |
| Zero-filled and mask-aware inputs | `src/eem_rc/data.py` | `eem_features.npy` |
| PLSR, RF, XGBoost, and zero-filled CNN benchmark | `src/eem_rc/evaluate.py` | `metrics_by_fold.csv`, `predictions.csv` |
| Mask-aware and masked-convolution sensitivity analysis | `src/eem_rc/models.py`, `src/eem_rc/evaluate.py` | LOSO metric tables and fold models |
| Full-data RCPM for interpretation | `src/eem_rc/interpret.py` | `rcpm_full_data.keras` |
| Grad-RAM at `conv2d_2` | `src/eem_rc/interpret.py` | `grad_ram.npy` |
| PCA to at most 100 components | `src/eem_rc/interpret.py` | `cnn_features_pca.npy` |
| MDA from normalized full-data predictions | `src/eem_rc/clustering.py` calling the separately installed `mda-learn` package | `mda_embedding.csv` |
| Ward clustering and K = 2–8 epsilon-squared comparison | `src/eem_rc/clustering.py` | `cluster_selection.csv` |
| C1–C3 ordered by mean reducing capacity | `src/eem_rc/clustering.py` | `clusters_k3.csv` |
| C1–C2 and C2–C3 generalized 2D-COS | `src/eem_rc/twodcos.py` | compressed synchronous/asynchronous maps |

## Validation boundary

The LOSO results produced by `02_evaluate.py` estimate predictive performance. The full-data model produced by `03_interpret.py` is used only for Grad-RAM, MDA, and clustering and must not be reported as an independent performance estimate.
