# Release checklist

- [x] Remove machine-specific absolute paths.
- [x] Match the revised manuscript's 81 × 81 non-interpolated input representation.
- [x] Keep predictive validation separate from the full-data interpretation fit.
- [x] Include all model hyperparameters stated in the manuscript.
- [x] Add fast synthetic-data tests and validate the 530-sample local dataset.
- [x] Exclude local data, trained models, caches, and generated outputs.
- [x] State that the dataset is not currently public and is available from the corresponding author upon reasonable request.
- [x] Add the MIT License with the 2026 Qun Xue copyright notice.
- [x] Preserve the separate `RCPM_PREDICT` and `MDA` environments used in the study.
- [x] Keep the upstream MDA implementation outside the MIT-licensed source tree and document its license.
- [ ] Add the article DOI and final citation when available.
- [ ] Run the complete workflow in a clean environment and archive the environment lock information.
- [ ] Review the proposed GitHub branch before merging into `main`.
