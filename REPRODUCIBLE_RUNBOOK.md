
# Reproducible runbook

## 1. Evidence level

Release 2.0.0 is a path-neutral, aggregate-only capsule.  It preserves the
frozen compiler package 0.4.0, the complete 38-stage Attempt-T execution,
independent conformance checks and the separately audited headed-stud
application projection.  Attempts A--S are excluded in full.

## 2. Environment

Python 3.12 is the frozen development interpreter family.  Install the
synthetic/compiler environment with:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements/requirements-v32-lock.txt
python -m pip install -r requirements/requirements-test.txt
```

DataSAIL 1.3.0 and a working SCIP backend are required only for its comparator
route.  The headed-stud adapter, evaluator and figure environments have
separate locks and should not be silently collapsed into one environment.

## 3. Verify the released methods and receipts

```bash
python -m pytest tests -q
python tools/audit_public_package_attempt_t.py .
```

The final command validates and syntax-parses the full public file set, checks
the SHA-256 ledger, Attempt-T
closure and retention receipts, all 38 stage projections, the primary-evidence
lock and the headed-stud projection.

The tests archived under `src/round_snapshot/method/tests/` are frozen source
from the controlled execution tree and retain its original directory contract.
They are provenance evidence, not the path-neutral public test entry point, and
must not be edited merely to make a relocated snapshot pass.

## 4. External headed-stud data

Download data only from the official records in `DATASETS_AND_LINKS.csv` and
keep archives and reconstructed rows outside the repository.  The verified
reconstruction route is:

```bash
cd src/headed_stud_adapter
python fetch_frozen_stud_parents.py
python overlap_audit.py
python apply_overlap_review.py
python build_adapters_public.py
python ../application_evaluator/run_public_evaluation.py \
  --adapter-root . \
  --method-root ../round_snapshot/method \
  --output-root ../../derived/headed_stud_evaluator \
  --reference-external-metrics ../../app
```

Generated rows, identifiers and fold assignments remain local and ignored.
The frozen audit matched all 48 external metric cells to a maximum absolute
difference of `7.105427357601002e-15`.

## 5. Frozen bindings

- protocol: `SAVP-CONFIRMATORY-V3.2`;
- compiler package: `0.4.0`;
- repository release: `2.0.0`;
- master seed: `2026071204`;
- stages: `38`;
- replicates per stage: `100`;
- seed-table SHA-256:
  `bcc4ec1c43f780bb64e417fce81f21df52db1202db8e7fdbe79104295c18d4ad`;
- freeze-manifest SHA-256:
  `0b6e3498b425b5cddb725bdc7b8cf3477ce8461f3677c3283de667c1a4464760`.

## 6. Interpretation boundaries

- Aggregate tables support only the prespecified compiler/synthetic claims.
- The headed-stud application retains its source-overlap quarantine and scope
  limits.
- Source-composition intervals are not population confidence intervals.
- A contract can diagnose unsupported deployment; it cannot create missing
  evidence.
- Semantic exactness is not claimed to guarantee lower predictive error.
- The prespecified negative model-selection result remains reported.
