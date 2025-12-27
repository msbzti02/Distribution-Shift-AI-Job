# Distribution Shift in the AI Job Market

A machine learning project demonstrating how ML models degrade in production due to **distribution shift**.

## Key Results

| Metric | Training | Deployment | Change |
|--------|----------|------------|--------|
| Accuracy | 99.6% | 75.2% | **-24.4%** |
| ROC-AUC | 0.999 | 0.933 | -0.066 |
| Time to Failure | — | 1 month | Immediate |

## What This Proves

1. **Distribution shift is real** — 24.4% accuracy drop upon deployment
2. **Failure is immediate** — No gradual decay warning period
3. **Statistical drift tests are insufficient** — PSI/KS showed 0 warnings
4. **Retraining strategies are essential** — Static models fail fast

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python run.py
```

## Sample Output

When you run `python run.py`, you'll see:

```
======================================================================
DISTRIBUTION SHIFT ANALYSIS WITH ROBUSTNESS TESTING
======================================================================

[PHASE 1] Data Preparation
--------------------------------------------------
Loaded 15000 records from 2 files
Date range: 2024-01-01 to 2025-04-30
Creating temporal splits...
Training: 2024-01-01 to 2024-06-30, 5662 samples
Deployment: 9338 samples

Monthly windows:
  2024-07: 959 samples
  2024-08: 958 samples
  2024-09: 895 samples
  2024-10: 949 samples
  2024-11: 922 samples
  2024-12: 959 samples
  2025-01: 964 samples
  2025-02: 840 samples
  2025-03: 922 samples
  2025-04: 909 samples

[PHASE 2] Feature Engineering
--------------------------------------------------
Salary band thresholds: [75000. 110000. 155000.]
Skill vocabulary size: 25
Features: 35
Training samples: 5662

[PHASE 3] Baseline Model Training
--------------------------------------------------
Training baseline model...
Training accuracy: 0.9963
Training ROC-AUC: 1.0000

[PHASE 4] Monthly Deployment Evaluation
--------------------------------------------------
Month       Samples   Accuracy        AUC       ECE
--------------------------------------------------
2024-07        959      0.753      0.938    0.021
2024-08        958      0.701      0.923    0.065
2024-09        895      0.737      0.927    0.038
2024-10        949      0.738      0.928    0.038
2024-11        922      0.745      0.937    0.029
2024-12        959      0.761      0.937    0.026
2025-01        964      0.744      0.935    0.030
2025-02        840      0.710      0.923    0.070
2025-03        922      0.721      0.925    0.050
2025-04        909      0.752      0.933    0.044

============================================================
ROBUSTNESS TESTING SUITE
============================================================

[1/5] Data Robustness Tests...
[2/5] Label Robustness Tests...
[3/5] Concept Robustness Tests...
[4/5] Temporal Robustness Tests...
[5/5] System-Level Tests...

============================================================
ROBUSTNESS SUMMARY
============================================================
  PSI warnings: 0
  KS test failures: 0
  Calibration OK: True
  Time to failure: 1 months
  Degradation: sudden

======================================================================
FINAL EXPERIMENT RESULTS
======================================================================

[PERFORMANCE METRICS]
   Baseline accuracy:      0.996
   Final month accuracy:   0.752
   Performance drop:       0.244
   Baseline AUC:           1.000
   Final month AUC:        0.933

[DRIFT DETECTION]
   Drift triggers:         0 / 10
   First failure month:    2024-07

[ROBUSTNESS SUMMARY]
   PSI warnings:           0
   KS test failures:       0
   Calibration OK:         True
   Time to failure:        1 months
   Degradation pattern:    sudden

======================================================================
[SUCCESS] Experiment completed successfully!
  Results saved to: outputs
======================================================================
```

## Output Files

After running, check the `outputs/` folder:

| File | Contents |
|------|----------|
| `experiment_summary.json` | Full metrics summary |
| `robustness_results.json` | Robustness test results |
| `drift_results.json` | PSI, KS, JS values per month |
| `monthly_metrics.json` | Accuracy, AUC, ECE per month |

### experiment_summary.json
```json
{
  "baseline": {
    "accuracy": 0.996,
    "roc_auc": 0.999
  },
  "final_month": {
    "accuracy": 0.752,
    "roc_auc": 0.933
  },
  "performance_drop": 0.244,
  "drift_triggers": 0,
  "robustness": {
    "psi_warnings": 0,
    "ks_failures": 0,
    "calibration_ok": true,
    "months_to_failure": 1,
    "degradation_pattern": "sudden"
  }
}
```

## Project Structure

```
├── run.py                 # Main entry point
├── config.py              # Configuration
├── data_processing.py     # Data loading & preprocessing
├── model.py               # XGBoost classifier
├── drift_detection.py     # PSI, KS, JS, skill churn
├── evaluation.py          # Metrics & evaluation
├── robustness_tests.py    # Testing suite
├── run_experiments.ipynb  # Interactive notebook
└── outputs/               # Results
```

## Methodology

- **Training**: January - June 2024 (5,662 samples)
- **Deployment**: July 2024 - April 2025 (10 monthly windows)
- **Task**: Salary band classification (4 classes)
- **Model**: XGBoost

## Technologies

Python 3.9+, XGBoost, scikit-learn, pandas, numpy, scipy

## License

MIT License
