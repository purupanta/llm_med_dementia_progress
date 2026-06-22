# Dementia Progression Prediction with LLM-Assisted Medication-State Features

## Project overview

`dementia_progression` is a longitudinal machine-learning pipeline for predicting next-visit dementia progression among Mild Cognitive Impairment (MCI) visits in the National Alzheimer’s Coordinating Center (NACC) investigator dataset. The project evaluates whether medication information from repeated medication-name fields can improve prognostic modeling when converted into structured, auditable medication-state representations.

The analytic framework compares clinical-only prediction models with models that add structured medication features, LLM-assisted medication-state features, optional BioClinicalBERT/SapBERT medication-text representations, and longitudinal trajectory features. The pipeline includes participant-level train-validation-test splitting, held-out evaluation, bootstrap confidence intervals, calibration assessment, decision-curve analysis, temporal validation, output auditing, manuscript-readiness checks, and secondary neuropathology anchoring.

Neuropathology variables are not used as prediction features. They are reserved only for secondary biological anchoring when available.

## Scientific purpose

The pipeline is designed to answer a prognostic modeling question. It evaluates whether routinely collected medication profiles contain participant-state information that improves prediction of next-visit dementia progression beyond standard clinical predictors.

Medication-state features should be interpreted as prognostic markers of participant state, treatment context, and clinical complexity. They should not be interpreted as causal medication effects.

## Current package status

This README consolidates the project documentation through the v2.14 package. The v2.14 update adds a launcher configuration-path guard. It does not change analytic modeling, Step 2 LLM dictionary logic, feature construction, output definitions, or manuscript-readiness gates.

Major retained capabilities include:

- refactored seven-step source-code architecture under `src/`;
- split YAML configuration files under `configs/`;
- raw-keyed LLM drug-dictionary mapping using stable SHA-256 keys;
- strict Step 2 medication-state audit and certification outputs;
- accuracy-preserving Step 2 runtime speedups;
- validation-selected final model selection;
- manuscript-readiness gate for production LLM claims;
- output-integrity audit scripts;
- synthetic smoke-test configuration;
- optional neural medication-text representation using frozen BioClinicalBERT and SapBERT encoders.

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
├── run_pipeline.sh                  # Root launcher
├── run_pipelines.sh                 # Main one-command launcher
└── VERSION.txt                      # Package version marker
```

## Data governance and restricted data warning

The full production run requires access to the NACC investigator CSV. NACC data are subject to NACC data-use terms and should not be committed to a public GitHub repository unless the applicable agreement explicitly permits it.

The repository should include source code, configuration files, documentation, tests, synthetic examples, and non-sensitive resources. Restricted raw data, private derived data, and participant-level outputs should be handled according to the relevant data-use agreement.

## Required input for production

Place the real NACC investigator CSV at:

```text
ip/data_nacc65/investigator_nacc65 w last visit.csv
```

Official NACC dictionary CSV files can be placed at:

```text
resources/nacc_official/*.csv
```

When available, these dictionary files are used to populate official column descriptions in Step 1 source-column audits.

## Installation

Create and activate a Python environment using your preferred `virtualenv` workflow, then install requirements from the project root.

```bash
python -m pip install -r requirements.txt
```

If pretrained medication-normalization or neural medication-text representation is enabled, also install:

```bash
python -m pip install -r requirements-pretrained.txt
```

The launcher can also attempt dependency preflight for pretrained components when enabled by configuration.

## One-command production execution

From the project root, run:

```bash
bash run_pipelines.sh
```

Equivalent supported commands include:

```bash
bash run_pipeline.sh configs/project.yaml
bash run_pipelines.sh --config configs/project.yaml
python src/run_pipeline.py --config configs/project.yaml
python -m src.run_pipeline --config configs/project.yaml
```

The v2.14 launcher resolves relative configuration paths against the packaged project root and reports a clear error if `configs/project.yaml` is missing.

## Synthetic smoke test

For a quick non-production validation run that avoids live Ollama and external pretrained model loading:

```bash
PIPELINE_CONFIG=configs/smoke_synthetic.yaml \
LLM_PROVIDER=mock \
NEURAL_TEXT_REPRESENTATIONS_ENABLED=false \
bash run_pipelines.sh
```

A smoke run verifies software execution and output integrity. It should not be used to support manuscript claims about live LLM-derived medication-state features.

## Configuration system

The project uses a split YAML configuration design.

```text
configs/
  _base.yaml                              # composition entry point
  _base_paths.yaml                        # input and output paths
  _base_runtime.yaml                      # seed, row-window, resume, progress, manuscript gate
  _base_llm_medication_state.yaml         # LLM/Ollama/cache settings
  _base_neural_text_representations.yaml  # BioClinicalBERT/SapBERT settings
  step1.yaml                              # source loading and cohort construction
  step2.yaml                              # medication features and LLM abstraction
  step3.yaml                              # splitting, model fitting, tuning
  step4.yaml                              # evaluation, calibration, DCA, temporal validation
  step5.yaml                              # prediction, reclassification, subgroup, importance
  step6.yaml                              # neuropathology anchoring
  step7.yaml                              # final reports and model bundle
  project.yaml                            # production profile
  smoke_synthetic.yaml                    # synthetic validation profile
```

`configs/project.yaml` inherits the base and step-specific configuration files. The configuration loader deep-merges dictionaries in order, so production or validation profiles can override default settings without editing every base file.

Recommended editing pattern:

| Change needed | Primary configuration file |
|---|---|
| Input CSV or output folder | `configs/project.yaml` or `configs/_base_paths.yaml` |
| Row-window testing or resume | `configs/project.yaml` or `configs/_base_runtime.yaml` |
| Source-column mapping and MCI cohort rules | `configs/step1.yaml` |
| Medication features, LLM, cache, neural settings | `configs/step2.yaml`, `_base_llm_medication_state.yaml`, `_base_neural_text_representations.yaml` |
| Train-validation-test split and model fitting | `configs/step3.yaml` |
| AUROC, average precision, Brier, calibration, DCA, temporal validation | `configs/step4.yaml` |
| Reclassification, subgroup performance, permutation importance | `configs/step5.yaml` |
| Neuropathology anchoring | `configs/step6.yaml` |
| Final reports and model bundle | `configs/step7.yaml` |

## Pipeline steps

The executable pipeline is organized into seven numbered steps.

| Step | Module | Purpose |
|---:|---|---|
| 1 | `src/pipeline_step1_source_cohort.py` | Load source data, apply row/column controls, remove invalid rows, create source audit, construct MCI next-visit cohort. |
| 2 | `src/pipeline_step2_medication_features.py` | Build clinical, structured medication, LLM-enhanced medication-state, neural medication, and longitudinal trajectory feature datasets. |
| 3 | `src/pipeline_step3_split_fit.py` | Create participant-level train-validation-test split, tune models, fit final models. |
| 4 | `src/pipeline_step4_evaluation.py` | Evaluate held-out performance, bootstrap CIs, calibration, decision curves, temporal validation, and feature importance. |
| 5 | `src/pipeline_step5_predictions.py` | Generate predicted-risk, reclassification, subgroup, and permutation-importance outputs. |
| 6 | `src/pipeline_step6_neuropathology.py` | Conduct secondary neuropathology anchoring when variables are available. |
| 7 | `src/pipeline_step7_reporting.py` | Write metadata, publication summaries, output synopses, final report, and model bundle. |

## Cohort and prediction target

The analytic task is next-visit dementia progression prediction among MCI index visits. Step 1 orders visits within participant, identifies eligible MCI visits with a subsequent visit, and labels whether dementia is observed at the next visit.

The pipeline uses required row identifiers such as `NACCID`, `NACCVNUM`, `VISITYR`, and `NACCUDSD` for downstream cohort construction and visit ordering. Missing clinical model covariates are not handled by complete-case deletion at Step 1. They are handled later by the modeling pipeline.

## Medication-state feature construction

Medication information is derived from repeated medication-name fields. The pipeline constructs several medication representations:

1. structured medication features derived from cleaned medication text and medication counts;
2. LLM-assisted medication-state domains from unique raw drug tokens;
3. optional frozen-encoder medication-text representation features using BioClinicalBERT and SapBERT;
4. longitudinal medication and clinical trajectory variables computed using current and prior visits only.

The LLM layer is used to convert medication names into structured medication-state descriptors. It is not used to predict dementia directly.

## LLM medication-state abstraction

LLM settings are stored primarily in:

```text
configs/_base_llm_medication_state.yaml
```

The production design uses a unique raw-drug dictionary. Each cleaned raw drug token receives a stable key:

```text
llm_dictionary_key = sha256(normalized raw drug token)
```

The LLM dictionary is then aggregated back to visit-level medication profiles by this stable raw-token key. Positional dictionary indexes are retained only for audit and are not used for downstream joining.

Important Step 2 LLM outputs include:

```text
s2_medication_state_features/s2e_unique_raw_drug_name_source.csv
s2_medication_state_features/s2e_unique_drug_name_llm_dictionary.csv
s2_medication_state_features/s2e_llm_medication_state_abstraction.csv
s2_medication_state_features/s2h_llm_medication_state_quality_audit.csv
s2_medication_state_features/s2k_llm_certification_audit.csv
s2_medication_state_features/s2p_llm_drug_dictionary_mapping_audit.csv
s2_medication_state_features/s2g_analysis_dataset_llm_enhanced_medication_state.csv
```

A row should be described as true LLM-derived only when the audit fields indicate successful live LLM parsing and appropriate provider status. Mock, fallback, or local abstraction rows are useful for software validation but should not be used to claim live LLM inference.

Structural certification means that the output follows expected parse and schema rules. It is not equivalent to expert clinical ground-truth validation.

## Ollama and local LLM runtime

The launcher can manage an isolated Ollama service for the dementia project while preserving a separate default Ollama service. The historical documentation describes use of an isolated port such as:

```text
127.0.0.1:11435
```

The default production profile expects an installed local instruction model according to the active YAML profile. Earlier documentation references Qwen2.5 profiles including `qwen2.5:7b-instruct` and failover candidates. Runtime model selection and failover behavior should be confirmed from the active `configs/_base_llm_medication_state.yaml` and launcher logs.

Useful stability overrides include:

```bash
LLM_MAX_CONCURRENT_REQUESTS=1 OLLAMA_NUM_PARALLEL=1 bash run_pipeline.sh configs/project.yaml
```

or, when GPU memory permits:

```bash
LLM_MAX_CONCURRENT_REQUESTS=2 OLLAMA_NUM_PARALLEL=2 bash run_pipeline.sh configs/project.yaml
```

## Optional BioClinicalBERT and SapBERT features

Optional neural medication-text representations are configured in:

```text
configs/_base_neural_text_representations.yaml
```

The supported frozen encoders include BioClinicalBERT and SapBERT. These models are not fine-tuned by the pipeline. They are used as frozen representation or normalization components when enabled and available.

Related outputs include:

```text
s2_medication_state_features/s2l_neural_text_representation_audit.csv
s2_medication_state_features/s2m_neural_medication_feature_summary.csv
s2_medication_state_features/s2n_analysis_dataset_neural_medication_state.csv
s3_model_training_validation/s3i_neural_representation_incremental_value.csv
```

For offline systems with local Hugging Face caches, run with local-file settings enabled in the relevant configuration or environment variables.

## Longitudinal trajectory features

The v2.8 extension adds leakage-aware longitudinal trajectory features prefixed with:

```text
traj_
```

These variables are computed within participant using only the current and prior visits. Future visits are never used to construct predictors for the current index visit.

The trajectory feature summary is written to:

```text
s2_medication_state_features/s2o_longitudinal_trajectory_feature_summary.csv
```

## Model development and validation

The main validation design uses participant-level splitting.

| Split | Purpose | Default requested proportion |
|---|---|---:|
| Training | Fit candidate models | 70% |
| Validation | Tune hyperparameters and fit calibration maps | 10% |
| Held-out test | Final untouched evaluation | 20% |

All visits from the same participant are assigned to one split only. The split audit is written to:

```text
s3_model_training_validation/s3a_participant_train_validation_test_split_summary.csv
```

The model evaluation framework includes:

- AUROC;
- average precision;
- Brier score;
- threshold-based metrics;
- bootstrap confidence intervals;
- calibration analysis;
- validation-set calibrated sensitivity analysis;
- decision-curve analysis;
- temporal validation;
- reclassification outputs;
- subgroup performance when minimum sample-size rules are met;
- permutation and grouped permutation importance when enabled.

Final predictive model selection can use validation-set AUROC with average precision and Brier score as secondary tie-breakers. Held-out test performance is not used for model selection.

## Temporal validation

Temporal validation is configured in Step 4. It evaluates whether models developed on earlier calendar-period visits maintain performance on later calendar-period visits. This analysis supports robustness assessment over calendar time and should be interpreted alongside the participant-level held-out test results.

## Calibration and absolute risk interpretation

Calibration outputs compare predicted and observed risk across risk strata and include validation-set calibrated sensitivity analyses when enabled. Discrimination metrics such as AUROC and average precision summarize ranking and precision-recall performance. Calibrated probabilities should be used for absolute risk interpretation when calibration mapping improves probabilistic prediction error.

## Decision-curve analysis

Decision-curve analysis estimates net benefit across prespecified threshold probabilities. It is used to evaluate clinical utility across plausible risk thresholds, not only discrimination.

## Neuropathology anchoring

Neuropathology variables are not model predictors. Step 6 uses neuropathology outcomes only as secondary anchoring variables when available. The purpose is to examine whether predicted clinical progression risk shows biological alignment with postmortem neuropathologic burden.

Related outputs include:

```text
s6_neuropathology_anchor_analysis/s6a_neuropathology_anchor_cohort.csv
s6_neuropathology_anchor_analysis/s6b_neuropathology_anchor_outcome_summary.csv
s6_neuropathology_anchor_analysis/s6c_neuropathology_anchor_risk_associations.csv
s6_neuropathology_anchor_analysis/s6d_neuropathology_anchor_risk_quartiles.csv
s6_neuropathology_anchor_analysis/s6e_neuropathology_anchor_interpretation.md
```

Neuropathology anchoring should be interpreted as supportive biological context, not as direct neuropathology prediction and not as evidence of medication causality.

## Runtime output layout

Each run writes to a timestamped folder under `op/`.

```text
op/dementia_progression_YYYYMMDD_HHMMSS/
```

Inside the run folder, outputs are organized by analytic step.

```text
logs/
s1_source_data_preparation/
s2_medication_state_features/
s3_model_training_validation/
s4_prediction_evaluation/
s5_publication_outputs/
s6_neuropathology_anchor_analysis/
s7_reporting_documentation/
```

A step folder is created only when at least one output file is written into that folder.

## Main output folders

| Folder | Role |
|---|---|
| `logs/` | Console and pipeline logs. |
| `s1_source_data_preparation/` | Source loading, row filtering, all-blank column removal, source audit, dictionary audit. |
| `s2_medication_state_features/` | MCI cohort summary and clinical, medication, LLM, neural, and trajectory feature datasets. |
| `s3_model_training_validation/` | Participant-level split, validation results, held-out test metrics, hyperparameter audit, incremental value tables. |
| `s4_prediction_evaluation/` | Risk scores, calibration, decision-curve analysis, temporal validation, reclassification, feature importance. |
| `s5_publication_outputs/` | Manuscript-ready model tables, narrative summary, trained model bundle, metadata. |
| `s6_neuropathology_anchor_analysis/` | Secondary neuropathology anchoring outputs when available. |
| `s7_reporting_documentation/` | Final output report, manuscript table index, output synopsis, project file synopsis. |

## Key output files

| Output file | Purpose |
|---|---|
| `s1_source_data_preparation/s1a_nacc_w_last_visit_filtered.csv` | Source dataset after raw-row selection and required-column row filtering. |
| `s1_source_data_preparation/s1b_incl_excl_cols.csv` | Source-column audit with included/excluded status, data type, official or fallback description, and dictionary provenance. |
| `s1_source_data_preparation/s1c_nacc_w_last_visit_filtered_final.csv` | Final downstream source after dropping entirely blank columns. |
| `s1_source_data_preparation/s1d_input_data_profile.csv` | Input profile documenting row-window selection and retained dimensions. |
| `s1_source_data_preparation/s1e_official_nacc_dictionary_download_audit.csv` | Official NACC dictionary auto-download audit when enabled. |
| `s2_medication_state_features/s2a_mci_next_visit_cohort_summary.csv` | MCI next-visit analytic cohort summary. |
| `s2_medication_state_features/s2b_clinical_only_feature_summary.csv` | Clinical-only feature summary with column-type labels. |
| `s2_medication_state_features/s2c_structured_medication_aware_feature_summary.csv` | Structured medication feature summary with column-type labels. |
| `s2_medication_state_features/s2d_analysis_dataset_structured_medication_state_features.csv` | Structured medication analysis dataset. |
| `s2_medication_state_features/s2e_llm_medication_state_abstraction.csv` | LLM medication-state abstraction audit. |
| `s2_medication_state_features/s2f_llm_enhanced_medication_feature_summary.csv` | LLM-enhanced medication feature summary. |
| `s2_medication_state_features/s2g_analysis_dataset_llm_enhanced_medication_state.csv` | LLM-enhanced medication analysis dataset. |
| `s2_medication_state_features/s2h_llm_medication_state_quality_audit.csv` | LLM quality-control and coverage audit. |
| `s2_medication_state_features/s2k_llm_certification_audit.csv` | Conservative structural certification audit. |
| `s2_medication_state_features/s2p_llm_drug_dictionary_mapping_audit.csv` | Stable-key mapping audit for raw-drug dictionary alignment. |
| `s3_model_training_validation/s3a_participant_train_validation_test_split_summary.csv` | Participant-level split and leakage checks. |
| `s3_model_training_validation/s3b_heldout_test_model_performance.csv` | Main held-out model performance table. |
| `s3_model_training_validation/s3c_heldout_test_model_performance_bootstrap_ci.csv` | Bootstrap confidence intervals. |
| `s3_model_training_validation/s3d_medication_incremental_value_vs_clinical_only.csv` | Medication-aware model deltas versus clinical-only models. |
| `s3_model_training_validation/s3f_validation_model_performance.csv` | Validation-set model performance. |
| `s3_model_training_validation/s3g_validation_hyperparameter_tuning_results.csv` | Hyperparameter tuning audit. |
| `s4_prediction_evaluation/s4a_heldout_test_predicted_risk_scores.csv` | Held-out visit-level predicted risk scores. |
| `s4_prediction_evaluation/s4d_heldout_test_calibration_by_decile.csv` | Calibration by predicted-risk decile. |
| `s4_prediction_evaluation/s4e_heldout_test_decision_curve_analysis.csv` | Decision-curve output. |
| `s4_prediction_evaluation/s4h_validation_calibrated_model_performance.csv` | Validation-calibrated model performance. |
| `s4_prediction_evaluation/s4l_temporal_validation_model_performance.csv` | Calendar-time temporal validation performance. |
| `s4_prediction_evaluation/s4m_validation_selected_final_model.csv` | Validation-selected final model audit. |
| `s5_publication_outputs/s5a_publication_model_performance_table.csv` | Manuscript-ready model performance table. |
| `s5_publication_outputs/s5b_publication_main_findings_summary.md` | Manuscript-oriented narrative findings summary. |
| `s5_publication_outputs/s5c_reproducible_trained_model_bundle.joblib` | Serialized fitted model bundle. |
| `s5_publication_outputs/s5d_run_metadata.json` | Machine-readable run metadata. |
| `s7_reporting_documentation/s7a_manuscript_table_index.csv` | Manuscript table index. |
| `s7_reporting_documentation/s7b_final_outputs_report.md` | Final run report. |
| `s7_reporting_documentation/s7c_output_file_synopsis.csv` | Machine-readable output synopsis. |
| `s7_reporting_documentation/s7d_project_file_synopsis.csv` | Machine-readable project file synopsis. |

## Resume and checkpointing

Resume can be controlled in YAML:

```yaml
runtime:
  resume:
    enabled: true
    output_dir: "./op/dementia_progression_YYYYMMDD_HHMMSS"
    use_latest_matching_output_dir: false
    allow_overwrite_existing_outputs: true
```

or from the shell:

```bash
DEMENTIA_RESUME_OUTPUT_DIR="./op/dementia_progression_YYYYMMDD_HHMMSS" bash run_pipelines.sh
```

Step 2 uses a persistent unique-drug dictionary cache:

```text
op/llm_medication_drug_dictionary_cache.csv
```

Partial Step 2 audit files are written during long LLM runs, allowing progress inspection before Step 2 completes.

## Manuscript-readiness gate

The project includes a strict manuscript-readiness gate:

```bash
python scripts/manuscript_readiness_gate.py <output_dir>
```

When enabled in the production configuration, the pipeline runs this gate after Step 7 and can fail the run if hard errors are detected.

The gate checks that required Step 2 LLM files exist, stable raw-drug keys align, downstream mapping uses `llm_dictionary_key`, visit rows align across key Step 2 datasets, prohibited supplement-label errors are absent, and the certification audit does not flag the output as unsuitable for LLM accuracy claims.

Production LLM claims should not be made unless the manuscript-readiness gate reports zero hard failures.

## Output-integrity audit

Completed run folders can be audited with:

```bash
python scripts/audit_outputs.py <output_dir>
```

This script checks required outputs and reports pass, warning, or failure statuses. Small smoke runs may produce expected warnings for empty neuropathology-anchor outputs if no eligible neuropathology rows are present.

## Validation history consolidated from documentation

The documentation records multiple validation checkpoints across versions. The most recent consolidated validation claims include:

- Python compile checks passed;
- shell syntax checks passed where applicable;
- unit tests passed in the assistant validation environment;
- synthetic seven-step smoke runs passed;
- synthetic output-integrity audits passed;
- real-input 1,000-row mock/local validation runs passed;
- real-input output-integrity audits passed with warnings limited mainly to empty neuropathology-anchor outputs in small subsets.

The assistant validation environment did not complete a full 188,700-row production run with live Ollama, GPU BioClinicalBERT, or SapBERT. Final manuscript AUROC, calibration, and LLM/neural performance claims must be based on the user’s full production outputs.

## Version history and key corrections

### v1.49 speed and resume design

Step 2 reduced unnecessary LLM generation by computing formula-derived medication-state fields in Python after parsing. Unique medication abstractions could be cached and resumed after interruption.

### v1.50 refactored GitHub-ready architecture

The pipeline was refactored into numbered modules under `src/`, while preserving the one-command launcher and validated analytic behavior.

### v1.51 split YAML configuration and NACC dictionary audit

Configuration was split into shared base YAML files and one YAML file per analytic step. Step 1 dictionary audit behavior was strengthened with official NACC dictionary description support when dictionary CSVs are available.

### v2.0 value-cleaning and dictionary provenance corrections

The v2.0 update preserved the v1.51 structure while strengthening Step 1 dictionary provenance, adding `column_type` labels to Step 2 feature summaries, and improving handling of structural missing-value codes.

### v2.1 CSV integer formatting and value audit correction

CSV export and audit behavior were corrected to preserve integer-like values and support safer inspection of analytic variables.

### v2.2 real-run analysis-clean guard

A fail-fast guard was added to prevent unsafe Step 2 analysis outputs from being written when critical value-cleaning problems are detected.

### v2.3 real-run Step 2 export finalization

Step 2 export checks were strengthened so final analysis datasets and feature summaries are written only after required checks pass.

### v2.4 column-specific `-4` sentinel fix

The project adopted column-specific handling for `-4` sentinel values rather than treating all appearances uniformly when inappropriate.

### v2.5 NACC special-code policy

Negative structural missing codes are treated as missing globally. Positive special codes are handled by official dictionary rules or prespecified variable-specific logic rather than by global blanket replacement.

### v2.6 and v2.7 medication-text and output-folder refinements

Empty or noninformative medication text was excluded from LLM abstraction, and runtime output folders were renamed to meaningful step-based folder names.

### v2.8 longitudinal trajectory features

Leakage-aware longitudinal trajectory variables were added using current and prior visits only.

### v2.11 command-line usability and raw-keyed LLM dictionary safety

Command-line invocation was simplified. The LLM drug dictionary was corrected to join by stable raw-token SHA-256 keys rather than positional row indexes or canonicalized text.

### v2.13 manuscript-readiness and accuracy-preserving speed controls

A strict manuscript-readiness gate, validation-selected final model logic, model-arm filtering for diagnostic runs, and runtime speedups were added without reducing raw-drug dictionary coverage or disabling strict LLM validation.

### v2.14 launcher config path guard

Launcher and pretrained preflight scripts now resolve config paths relative to the project root and produce a clear error when the requested config file is missing.

## GitHub upload guidance

Recommended files to include in GitHub:

```text
configs/
docs/
helpers/
ip/synthetic/
resources/medication/
resources/nacc_official/README.md
scripts/
src/
tests/
README.md
LICENSE
requirements.txt
requirements-pretrained.txt
run_pipeline.sh
run_pipelines.sh
VERSION.txt
```

Usually exclude virtualenv runtime files and machine-specific artifacts:

```text
bin/
share/
pyvenv.cfg
CACHEDIR.TAG
__pycache__/
.ipynb_checkpoints/
```

Do not commit restricted raw NACC data or private participant-level outputs unless explicitly permitted.

## Recommended root `.gitignore`

```gitignore
# virtualenv files/folders created at project root
/bin/
/share/
/pyvenv.cfg
/CACHEDIR.TAG
/lib/
/include/
/Scripts/
/Lib/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Jupyter temporary files
.ipynb_checkpoints/

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/

# Logs/temp files
*.log
*.tmp
```

This `.gitignore` preserves project files while excluding virtualenv and machine-generated files. Do not use a root `.gitignore` containing only `*`, because that ignores the entire project.

## Interpretation boundaries

This repository provides a reproducible modeling and audit pipeline. It does not establish medication causality, expert-validated medication classification truth, or clinical deployment readiness by itself.

Manuscript claims should be based on completed production outputs, including:

- held-out test performance;
- bootstrap confidence intervals;
- temporal validation;
- calibration outputs;
- decision-curve analysis;
- LLM certification and mapping audits;
- manuscript-readiness gate results;
- secondary neuropathology anchoring, when available.

External validation, ontology-based medication adjudication, expert review of medication-state classifications, subgroup calibration assessment, and prospective evaluation remain necessary before clinical implementation.

## Source documentation consolidated

This README consolidates the following Markdown files from `docs/`:

```text
CONFIGURATION_GUIDE.md
OUTPUT_FILE_SYNOPSIS.md
PROJECT_FILE_SYNOPSIS.md
README.md
REFACTORING_AND_GITHUB_GUIDE.md
VALIDATION_SUMMARY.md
V1_50_REFACTOR_VALIDATION_SUMMARY.md
V1_51_STEP_YAML_NACC_DICTIONARY_VALIDATION_SUMMARY.md
V2_0_ANALYSIS_VALUE_CLEANING_CORRECTION.md
V2_0_CORRECTION_AND_REPACKAGE_SUMMARY.md
V2_0_NPPMIH_STRUCTURAL_MISSING_CORRECTION.md
V2_0_PACKAGE_RECHECK_SUMMARY.md
V2_1_CSV_INTEGER_FORMAT_AND_VALUE_AUDIT.md
V2_2_REAL_RUN_ANALYSIS_CLEAN_GUARD.md
V2_3_REAL_RUN_EXPORT_FINALIZATION.md
V2_4_COLUMN_SPECIFIC_MINUS4_SENTINEL_FIX.md
V2_5_NACC_SPECIAL_CODE_POLICY.md
V2_6_EMPTY_MEDICATION_TEXT_EXCLUSION.md
V2_7_MEANINGFUL_OUTPUT_FOLDER_NAMES.md
V2_8_LONGITUDINAL_TRAJECTORY_FEATURES_AUROC_IMPROVEMENT.md
V2_9_COMMAND_LINE_USABILITY_FIX.md
V2_11_NACC_AUTODOWNLOAD_AND_RAW_KEYED_LLM_DICTIONARY_FIX.md
V2_11_VALIDATION_SUMMARY.md
V2_12_MANUSCRIPT_READINESS_SPEED_AND_VALIDATION_GATE.md
V2_13_ACCURACY_PRESERVING_SPEEDUPS.md
V2_14_LAUNCHER_CONFIG_PATH_GUARD.md
```
