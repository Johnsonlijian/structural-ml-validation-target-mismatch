# Headed-stud local reconstruction adapter

This directory reconstructs the four canonical headed-stud tables from the
versioned third-party archives listed in `freeze_v1.json`. Raw archives,
extracted rows, citation/source mappings, targets and canonical tables remain
local and are excluded by the repository `.gitignore`.

Run the steps in order from this directory:

```bash
python fetch_frozen_stud_parents.py
python overlap_audit.py
python apply_overlap_review.py
python build_adapters_public.py
```

The first command verifies upstream archive sizes and hashes. The next two
commands reconstruct the outcome-blind citation/source audit and apply the
frozen manual citation decisions. `build_adapters_public.py` verifies those
stable inputs before creating `derived_private/*.csv` and
`adapter_manifest.json` locally. Do not commit any generated row-level file.

Then reproduce the aggregate application evaluator:

```bash
python ../application_evaluator/run_public_evaluation.py \
  --adapter-root . \
  --method-root ../round_snapshot/method \
  --output-root ../../derived/headed_stud_evaluator \
  --reference-external-metrics ../../app
```

Strict mode reruns DataSAIL in memory and stops if no complete assignment is
returned. The evaluator persists aggregate metrics, typed statuses, fold
counts and hashes only; it does not persist targets, predictions, source/row
identifiers or assignment vectors. This reconstructs the headed-stud
application evaluator, not the 38-stage synthetic confirmatory pipeline.

The four upstream datasets retain their own licences. In particular, the RAC
parent and its aggregates remain CC BY-NC-SA 4.0 and must not be moved into a
permissively licensed data subtree.
