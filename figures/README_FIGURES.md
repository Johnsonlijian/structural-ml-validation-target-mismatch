## Figure inventory

| Target asset | Expected generator | Primary table / data |
|---|---|---|
| Figure 1 taxonomy + workflow | `code/26_make_figure1_study_design.py` (or final layout tool) | Methods taxonomy |
| Figure 2 heterogeneous gaps | `code/18_make_figure2_v2_uncertainty.py` | `outputs/figures_source/fig02_*` |
| Figure 3 metadata audit | literature QA figures | manual verification CSV |
| Figure 4 checklist | reporting checklist tooling | `manuscript/tables/reporting_checklist_v0.csv` |
| Figure 5 Mendeley joints | `code/25_make_mendeley_figure5_and_tables.py` | Mendeley reproduction outputs |
| Figure S1 Stub-CFST | `code/14_reproduce_stub_cfst_scirep_2024.py` | stub CFST reproduction |
| Figure S2 DesignSafe columns | `code/28_reproduce_designsafe_rc_columns.py` | DesignSafe column CSVs |
| Figure S3 PRJ-2430 walls | `code/30_reproduce_designsafe_prj2430_wall.py` | wall extraction CSV |
| Figure S4 PRJ-3053 | `code/32_reproduce_designsafe_prj3053_coupling_beams.py` | coupling-beam extraction |
| Figure S5 RF gap atlas | `code/33_cace_nine_module_summary.py` | `cace_ninemodule_rf_summary.csv` |
| Graphical abstract | `code/34_graphical_abstract_cace_v0.py` | same as S5 summary |

If a raster/vector file is missing from this archive, regenerate locally after datasets are downloaded.
