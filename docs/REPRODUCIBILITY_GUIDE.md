## Quick start

```bash
conda env create -f environment_reproducibility_v0.yml
conda activate p1-structural-ml-validation
python run_qa_chain.py --help
python run_qa_chain.py --smoke-test
```

Full reproduction requires downloading third-party datasets per `data_manifests/dataset_manifest.csv`.
