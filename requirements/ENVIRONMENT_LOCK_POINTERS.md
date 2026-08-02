# Environment lock pointers

The frozen v3.2 environment was captured with Python 3.12 and `pip freeze --all`.

- Environment-lock SHA-256: `5568b7ef34ab79a33d3f1c01401706e1a0c5dd1bdd993fcc588e90ea5fb4bae5`
- DataSAIL version: `1.3.0`
- Frozen DataSAIL wheel SHA-256: `7c6bed9c43e8f35f1e084db21835cfcaa9f862fff823fef68aa9dd3a34be3e13`
- NumPy: `1.26.4`
- pandas: `3.0.3`
- scikit-learn: `1.9.0`
- SciPy: `1.17.1`
- CVXPY: `1.5.3`
- PySCIPOpt: `6.2.1`

The controlled builder copies the verified text lock to
`requirements/requirements-v32-lock.txt` only after the 38-stage aggregate
gate passes. The vendor wheel is not copied because its redistribution and
dependency-licence bundle require a separate review. Install DataSAIL from an
authorized distribution channel and verify the version and solver backend.
