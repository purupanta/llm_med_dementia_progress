# Dementia Progression Prediction with LLM-Assisted Medication-State Features

## Project overview

`dementia\_progression` is a longitudinal machine-learning pipeline for predicting next-visit dementia progression among Mild Cognitive Impairment (MCI) visits in the National Alzheimer’s Coordinating Center (NACC) investigator dataset. The project evaluates whether medication information from repeated medication-name fields can improve prognostic modeling when converted into structured, auditable medication-state representations.

The analytic framework compares clinical-only prediction models with models that add structured medication features, LLM-assisted medication-state features, optional BioClinicalBERT/SapBERT medication-text representations, and longitudinal trajectory features. The pipeline includes participant-level train-validation-test splitting, held-out evaluation, bootstrap confidence intervals, calibration assessment, decision-curve analysis, temporal validation, output auditing, manuscript-readiness checks, and secondary neuropathology anchoring.

Neuropathology variables are not used as prediction features. They are reserved only for secondary biological anchoring when available.

## Scientific purpose

The pipeline is designed to answer a prognostic modeling question. It evaluates whether routinely collected medication profiles contain participant-state information that improves prediction of next-visit dementia progression beyond standard clinical predictors.

## Repository structure

```text
.
├── configs/                         # YAML configuration files
├── docs/                            # Original detailed documentation and validation notes
├── helpers/                         # Shared helper modules
├── ip/                              # Input folder; synthetic example data may be included
├── notebooks/                       # Optional analysis/plotting notebooks, when present
├── op/                              # Runtime output folder, when generated
├── resources/                       # Medication resources and optional NACC dictionary files
├── scripts/                         # Preflight, audit, and manuscript-readiness scripts
├── src/                             # Seven-step pipeline implementation
├── tests/                           # Synthetic data generator and unit tests
├── requirements.txt                 # Core Python requirements
├── requirements-pretrained.txt      # Optional pretrained encoder requirements
├── run\_pipeline.sh                  # Root launcher
├── run\_pipelines.sh                 # Main one-command launcher
└── VERSION.txt                      # Package version marker
```

