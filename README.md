# Stability of feature-attribution explanations for heart-disease risk models

Code and results for the paper *Stability of feature-attribution explanations for heart-disease
risk models: an audit across models, explainers, training samples and measurement noise*
(Yash Tyagi).

The study asks whether the explanation given for one patient (the three features ranked highest
by SHAP, Integrated Gradients or Expected Gradients) is a property of the patient, or of
incidental modelling choices. Four model classes (logistic regression, random forest, XGBoost,
a small neural network) are trained on two public datasets and their explanations are compared
across explainers, model classes, random seeds, bootstrap resamples of the training data,
equally good hyperparameter settings, measurement noise and background samples.

## Branches and tags

| Name | Content |
| --- | --- |
| `main` | This study (code, protocol, results, summary) |
| `protocol-v1` | Same history as `main`; the branch named in the paper |
| tag `protocol-v1.0` | The protocol as frozen before any reliability score was computed |
| `original-draft` | The earlier draft's notebook, figures and README, kept unchanged |

Start with `summary.txt` for a plain-language account of the whole study.

## Layout

| Path | Content |
| --- | --- |
| `Methodology_and_Data_Flow.txt` | Protocol (frozen as tag `protocol-v1.0`) and its amendment log |
| `Decision_Register.txt` | Every design decision, its alternatives and the supporting literature |
| `Validation_and_Defence.txt` | Audit of the design: what was checked, what changed, what remains limited |
| `short_summary_of_work.txt` | Plain-language walk-through with data-flow diagrams |
| `src/` | Analysis code (see below) |
| `kaggle/` | Runners used for the heavy jobs on Kaggle CPU kernels |
| `results/` | Reports and tables produced by the code (model files and per-variant arrays are not stored) |
| `legacy/` | Notebook and figures of an earlier draft. Not used by the paper |

## Data

The two CSV files are not redistributed here. Place them in `data/`:

| File | Source | SHA-256 |
| --- | --- | --- |
| `framingham.csv` (4,240 x 16) | Public Framingham teaching extract, mirror `GauravPadawe/Framingham-Heart-Study` on GitHub | `2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018` |
| `heart_disease_health_indicators_BRFSS2015.csv` (253,680 x 22) | Kaggle, A. Teboul, "Heart Disease Health Indicators Dataset" | `953fb23bb131736d7879539a071d3f6d1eaae8d4867ab05335d0ba89b94d5f99` |

`kaggle/cdc_check.py` rebuilds the BRFSS file from the CDC 2015 raw file and compares it with the
public file (report in `results/brfss/cdc_check_report.txt`).

## Environment

Python 3.13 with the versions in `requirements.txt`. Everything runs on CPU. Framingham runs on a
laptop (8 GB RAM); BRFSS needs about 16 GB and was run on Kaggle CPU kernels.

## Reproducing

Paths are set by environment variables: `XAI_DATA` (default `data/`), `XAI_OUT` (default
`results/`), `XAI_SEED` (master seed, default 42; controls the split, the explained patients, the
background rows, tuning folds and model seeds).

```
python src/phase1_data_validation.py
python src/phase2_reference.py
python src/phase3_models.py <dataset> base
python src/phase3_models.py <dataset> variants
python src/phase5_multiplicity.py <dataset>
python src/phase4_base_axes.py <dataset> faith|noise|floor|kernel
python src/phase5_plausibility.py                      # Framingham
python src/phase6_variance.py <dataset> noise|background|report
python src/phase6_shares_linking.py
python src/phase6_smote_arm.py <dataset>
python src/phase6_sample_size.py run <n> [seed]        # BRFSS
python src/phase6_sample_size.py report
```

`<dataset>` is `framingham` or `brfss`. Long stages are resumable and stop after a time budget
(`XAI_BUDGET`, seconds).

Checks on the design:

| Script | What it checks |
| --- | --- |
| `audit1_correctness.py` | Explainers against exact Shapley values; additivity; metric code |
| `audit2_sensitivity.py` | Thresholds, complete cases, grouping, near-ties |
| `audit3_alternatives.py` | Other splits, re-tuning inside bootstraps, imputation, path-dependent TreeSHAP, subsampling |
| `audit4_settings.py` | Settings fixed by choice (forest size, learning rate, network width, optimiser, wider grids) |
| `audit5_stored.py`, `audit7_stored.py` | Analysis settings re-examined from stored explanations |
| `audit6_hyperparameters.py` | Equally good hyperparameter settings as a source of variation |

Replication under other master seeds: `kaggle/make_jobs.py` writes the Kaggle kernels;
results are in `results/replication_seed7/` and `results/replications/`.

## Licence

See `LICENSE`.
