# Reproducibility materials for the IJFS revision

**Study:** *Distribution-Driven Linearization of Credibilistic Mean-RVaR Portfolio Optimization under Coherent Trapezoidal Fuzzy Returns*

This repository contains the Python implementation, archived empirical inputs and outputs, and checks associated with the Model A (minimum-downside) portfolio experiments. It distinguishes three activities: (i) inspecting previously computed results; (ii) independently rebuilding Stage 1 from local BaoStock caches; and (iii) rerunning Stages 2–5 and completing supplementary diagnostics. A successful integrity check is **not** equivalent to an independent recomputation or to certification of live market fills.

## 1. Repository layout

All commands below are run from the directory containing `step1.py` (the top-level `IJFS/` directory in the distributed archive). The key contents are:

```text
IJFS/
├── README.md                       # This document
├── README_zh.md                    # Chinese operation notes
├── requirements.txt               # Python package requirements
├── CODE_SHA256_final.json               # Archived source hashes; see version note below
├── data/                          # Constituent history, bond-yield and index data
├── results1/                      # Archived Stage 1 outputs AND baostock_cache/
│   ├── baostock_cache/            # Local stock-price and turnover CSV caches
│   ├── full_return_matrix.csv
│   ├── full_turnover_matrix.csv
│   ├── risk_free_rate.csv
│   ├── rolling_windows.csv
│   ├── windows_universe.csv
│   ├── fuzzy_parameters.csv
│   ├── endpoint_calibration.csv
│   ├── bootstrap_indices.npz
│   └── step1_run_metadata.json
├── results_execution/             # Archived completed calculations
│   ├── results2/                  # Optimization and locked policy
│   ├── results3/                  # Sequential backtests and holdings
│   ├── results4/                  # Performance and statistical inference
│   ├── results5/                  # Generated figures
│   ├── results5_0/                # Additional historical figure directory
│   ├── results_remaining/         # Remaining reviewer diagnostics and shape fits
│   └── completion/                # Existing completion and execution checks
├── manuscript/                    # Manuscript, supplementary TeX, bibliography, figures
├── step1.py ... step5.py          # Five core stages
├── baseline_pipeline.py           # Stages 2–5 + remaining experiments
├── finish_all.py                  # Post-processing of an existing computed run
├── run_all.py                     # Calls baseline_pipeline then finish_all, or completes --source
├── empirical_statistics.py        # HAC, block-bootstrap, Holm and other statistics
├── remaining_experiments.py       # Controlled quantile path and endpoint resampling
├── ki_calibration.py              # Conditional asset-specific shape projection
├── review_completion.py          # Reviewer-oriented results consistency checks
├── controlled_diagnostics.py      # Controlled supplementary comparisons
├── execution_sensitivity.py      # Execution-timing/fallback sensitivity
├── data_evidence_audit.py        # Local-cache/source reconciliation
└── test_*.py / test_support*.py   # Regression, mathematical and synthetic tests
```

The archive may also contain legacy drafts and historical output files. **Do not assume that every file under `manuscript/` is the current submission version.** Match the submitted TeX/PDF and figures explicitly. The archive's `CODE_SHA256_final.json` must be compared against the exact distributed code version before it is used as evidence of provenance; source edits change hashes.

## 2. Environment

The experiments are Python-based. In a terminal opened at the repository root:

```powershell
python -m pip install -r requirements.txt
python --version
python -m pip freeze > environment_installed.txt
```

The requirements include NumPy, pandas, SciPy, PyArrow, CVXPY, OSQP, Matplotlib and openpyxl. They specify minimum versions rather than a byte-identical environment lock. One archived run records Python 3.9.23 / SciPy 1.13.1. For numerical comparisons, also record the operating system and LP/QP solver versions. `environment_installed.txt` is generated locally; it is not an input in the supplied archive.

## 3. Required execution order and Stage 1

**Stage 1 must be run separately before a genuinely fresh Stages 2–5 recomputation.** It reads `results1/baostock_cache/` and the files in `data/`, builds daily return/turnover matrices, aligns the bond-rate proxy, creates the 40 rolling-window and eligibility records, calibrates fuzzy endpoints, and generates frozen bootstrap indices. The cache build does not download data unless the user explicitly requests `--allow-network`.

The default design uses 250 estimation days, 60 holding days, a 60-day step, 40 windows, at least 200 usable estimation returns per candidate, baseline endpoint probabilities (0.05, 0.25, 0.75, 0.95), alternative endpoint grids, and 300 draws per 5-day and 10-day resampling scheme.

### 3A. Fresh copy in which `results1/` contains the cache but **no frozen Stage 1 core files**

In this case, `step1.py` can populate `results1/` directly:

```powershell
python step1.py --mode build --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\results1
```

This route is valid **only if** the frozen Stage 1 core CSVs are not already present in `results1/`. The code deliberately rejects an in-place rebuild of a populated baseline, even with `--overwrite`.

### 3B. This supplied archive: `results1/` already contains frozen Stage 1 outputs

To **use the frozen outputs**, do not rebuild Stage 1. See §4A below. To **independently verify the cache-to-Stage-1 computation**, direct the build to a new, previously empty output location (the name is illustrative and is created on execution):

```powershell
python step1.py --mode build --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\stage1_independent
```

`stage1_independent/` is **not a pre-existing input folder**. It is the output of this optional rebuild. It must not be mixed with existing `results_execution/` outputs without checking the Stage 1 provenance and hashes. To audit the archived Stage 1 without recreating the core matrices:

```powershell
python step1.py --mode audit --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\stage1_audit
```

The audit-only directory is likewise an output, not part of the original folder tree. Do not use `--skip-bootstrap` for a fresh full experiment, because subsequent endpoint diagnostics require `bootstrap_indices.npz`.

**Stage 1 acceptance:** check `step1_run_metadata.json` (successful status, 40 completed windows, no failed integrity checks), `input_data_audit.csv`, `step1_run.log`, and the availability of `full_return_matrix.csv`, `full_turnover_matrix.csv`, `risk_free_rate.csv`, `rolling_windows.csv`, `windows_universe.csv`, `fuzzy_parameters.csv`, `endpoint_calibration.csv`, and `bootstrap_indices.npz`. Review flags are not automatically errors, but must not be silently discarded. A failed Stage 1 build is not a valid input to Stages 2–5.

## 4. Choose one workflow

### 4A. Post-process the supplied archived run (no main experiment recomputation)

The archive already has a complete `results1/` and a `results_execution/` parent containing results from Stages 2–5. To perform the existing-run completion checks without replacing the archive:

```powershell
python finish_all.py --source .\results_execution --input .\results1 --data .\data --cache .\results1\baostock_cache --out .\completion_submission_check
```

This command expects `--source` to point to the **parent** of `results2/`, `results3/`, and `results4/`, not to its `completion/` subfolder. The `--out` destination must be new or empty; it is not one of the archived directories. This workflow does not rerun the main rolling optimization, and should not be described as an independent end-to-end reproduction.

### 4B. Recompute the full experiment from local caches

First perform **§3A or §3B Stage 1 build**. If §3B was used and the new Stage 1 output passed acceptance, run:

```powershell
python baseline_pipeline.py --input .\stage1_independent --data .\data --out-root .\modelA_independent
python finish_all.py --source .\modelA_independent --input .\stage1_independent --data .\data --cache .\results1\baostock_cache --out .\completion_independent
```

If §3A populated `results1/` directly, replace `stage1_independent` by `results1` in these two commands. `baseline_pipeline.py` invokes Stage 2, Stage 3, remaining reviewer experiments, Stage 4, and Stage 5 in that order. It writes `results2/`, `results3/`, `results_remaining/`, `results4/`, `results5/`, and a completion checklist beneath the output root. Formal defaults require at least 2,000 paired bootstrap replications and exactly 300 frozen endpoint draws; pilot results are **not** publication evidence.

`run_all.py` is an alternative orchestrator **after Stage 1 is available**; it is not a Stage 1 builder. For a fresh run, `python run_all.py --input .\stage1_independent --data .\data --out-root .\modelA_unified` invokes `baseline_pipeline` and then `finish_all`. For completion of archived results, use `--source .\results_execution` with a fresh `--out-root`.

### Optional execution-policy sensitivity

Add `--with-execution` to `finish_all.py` when full Stage 1, the complete Stage 2 directory and index data are available. It runs formation-close repair, next-close hold and next-close repair alternatives, along with their own downstream statistics and charts. These idealized close-value/turnover-proxy variants are **not** observed order-level fills. For example:

```powershell
python finish_all.py --source .\modelA_independent --input .\stage1_independent --data .\data --cache .\results1\baostock_cache --out .\completion_execution --with-execution
```

The provided `finish_all.py` does **not** accept `--verify-only`, `--overwrite` or `--endpoint-replicates`; `--pilot` and `--replicates` are supported. Do not execute outdated commands from earlier draft instructions.

## 5. Key model and reporting conventions

- Decision model: **Model A**, minimum net downside risk subject to a required net credibilistic return; no scalarized mean reward in the objective.
- Equal purchase/sale transaction costs: **0.30% per side** (`0.003`), financed by the portfolio's cash budget; the weights and constraints use pre-trade wealth units.
- Rolling execution: each strategy retains its own positions and cash and carries mark-to-market drift; numerical infeasibility leads to the disclosed no-trade hold rather than a fictitious optimized allocation.
- Main criteria: return-side CVaR lower tail 0.05, RVaR interval [0.05, 0.20], and CVaR lower tail 0.20. A same-return-object quantile identity does not imply equal outcomes after separate portfolio optimizations.
- Full-policy results include fallback windows; Direct-only samples have their own eligibility rules. Do not splice nonconsecutive dates together when computing drawdown.
- Statistical output uses HAC inference, paired block bootstrap and Holm family adjustment as implemented in the source. The inferential family and sample scope must be kept with reported p-values.

## 6. Outputs, validation and limitations

| Directory or file | Role |
|---|---|
| `results1/` | Frozen Stage 1 matrices, rolling design, endpoints, resampling indices and input audit |
| `results_execution/results2/` | Optimization specifications, scenario outputs and `lp_policy_lock.json` |
| `results_execution/results3/` | Daily realized returns, window actions, holdings and execution status |
| `results_execution/results4/` | Performance tables, paired comparisons, statistical and market-state diagnostics |
| `results_execution/results5/` | Generated graphics, to be matched against the submitted `manuscript/figures/` |
| `results_execution/results_remaining/` | Endpoint uncertainty, controlled comparisons and conditional shape projections |
| `results_execution/completion/` | Previously archived reviewer checks and execution sensitivity variants |
| `reviewer_evidence_checklist.csv`, `finish_metadata.json` | New completion-run records in the selected `--out` directory |

Run synthetic/regression tests from the repository root:

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

Tests check program invariants and fixtures; they do not establish that the historical main results have been independently recomputed. If a module is unavailable, first check installation and whether the complete source archive was supplied. Do not change reference results or assertions merely to obtain a passing test. Validate final code and output SHA-256 values against the exact version being submitted; changes to Python files invalidate older source hashes and metadata.

**Provenance and sharing restrictions.** Reconstructing matrices from supplied local BaoStock caches does not independently verify the original download receipts, adjustment flags, corporate-action ledger, historical index membership publication dates or actual transaction fills. A positive turnover observation is an execution-eligibility proxy, not an exchange fill report. Third-party raw-market-data redistribution is conditional on the source's terms; if redistribution is not permitted, supply input specifications and permitted derived artifacts instead and disclose the limitation. A public repository or journal attachment identifier must be stated **only after actual deposition**. The presence of code and data in a local ZIP does not prove external accessibility or all reviewer concerns resolved.

## 7. Submission checklist

Before depositing the code archive: (1) identify one final source-code version; (2) ensure that final `main.tex`, `supplement.tex`, bibliography and figures match the PDF versions; (3) run available tests and record environment and failures honestly; (4) verify provenance hashes and the claimed source/results correspondence; (5) include permitted derived outputs and the explicitly cited statistical CSV files; and (6) align the paper's Data/Code Availability statements with the actual archived files and verifiable deposit location.

This README documents how to use the submitted materials. It does not certify that every environment-dependent recomputation or external-data provenance check has passed.


### Final validation and source hashes (2026-11-06)

`CODE_SHA256_final.json` records SHA-256 digests of the 27 primary Python source and test files in this archive. The historical `validation_final/TEST_SOURCE_SHA256_final.json` records the exact broader project snapshot at the time of the 106/106 passing test run, **including temporary patch and backup files that have since been removed from this cleaned distribution**; it is kept as a provenance record and is not intended to match every file in the cleaned archive. The successful `validation_final/VALIDATION_final.json` reports 106 tests passing with return code 0. These automated tests do not represent a fresh full-data experimental rerun (`real_full_data_rerun: false`).

The final code, tests, and scientific results were not modified by this archive clean-up; only one-off installers, old backup copies, obsolete manifests, Python caches, and these README references were cleaned. When retesting this cleaned distribution, use `python -m unittest discover -s . -p "test_*.py" -v` from the project root.
