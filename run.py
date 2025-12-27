import numpy as np
import pandas as pd
from pathlib import Path
import json
from config import MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR
from data_processing import load_and_prepare_data, FeaturePreprocessor, get_temporal_splits, validate_no_leakage
from model import BaselineModel, train_baseline
from drift_detection import DriftDetector
from evaluation import calculate_all_metrics, MonthlyEvaluator
from robustness_tests import run_all_robustness_tests, save_robustness_results

def main():
    print("=" * 70)
    print("DISTRIBUTION SHIFT ANALYSIS WITH ROBUSTNESS TESTING")
    print("=" * 70)
    print("\n[PHASE 1] Data Preparation")
    print("-" * 50)
    
    df = load_and_prepare_data()
    train_df, monthly_windows = get_temporal_splits(df)
    validate_no_leakage(train_df, monthly_windows)
    print("\n[PHASE 2] Feature Engineering")
    print("-" * 50)
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(train_df)
    X_train, y_train = preprocessor.transform(train_df)
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print("\n[PHASE 3] Baseline Model Training")
    print("-" * 50)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = train_baseline(X_train, y_train, preprocessor.feature_columns)
    y_train_proba = model.predict_proba(X_train)
    baseline_metrics = calculate_all_metrics(y_train, model.predict(X_train), y_train_proba)
    print("\n[PHASE 4] Monthly Deployment Evaluation")
    print("-" * 50)
    evaluator = MonthlyEvaluator()
    evaluator.set_baseline(baseline_metrics)
    drift_detector = DriftDetector()
    drift_detector.set_reference(
        X_train, y_train_proba, preprocessor.feature_columns,
        train_df['required_skills'] if 'required_skills' in train_df.columns else None
    )

    print(f"\n{'Month':<10} {'Samples':>8} {'Accuracy':>10} {'AUC':>10} {'ECE':>8}")
    print("-" * 50)

    for month_label, month_df in monthly_windows:
        X_month, y_month = preprocessor.transform(month_df)
        metrics = evaluator.evaluate_month(model, X_month, y_month, month_label)
        skills = month_df['required_skills'] if 'required_skills' in month_df.columns else None
        drift_detector.detect(X_month, model.predict_proba(X_month), month_label, skills)
        
        print(f"{month_label:<10} {metrics['n_samples']:>8} {metrics['accuracy']:>10.3f} "
              f"{metrics['roc_auc']:>10.3f} {metrics['calibration']['ece']:>8.3f}")


    robustness_results = run_all_robustness_tests(
        model=model,
        preprocessor=preprocessor,
        baseline_metrics=baseline_metrics,
        train_df=train_df,
        monthly_windows=monthly_windows,
        drift_history=drift_detector.drift_history,
        monthly_metrics=evaluator.monthly_results
    )
    
    save_robustness_results(robustness_results)
    trends = evaluator.get_trends()
    drift_summary = drift_detector.get_summary()
    robust_summary = robustness_results['summary']

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 70)
    
    print("\n[PERFORMANCE METRICS]")
    print(f"   Baseline accuracy:      {baseline_metrics['accuracy']:.3f}")
    print(f"   Final month accuracy:   {trends['accuracy'][-1]:.3f}")
    print(f"   Performance drop:       {baseline_metrics['accuracy'] - trends['accuracy'][-1]:.3f}")
    print(f"   Baseline AUC:           {baseline_metrics['roc_auc']:.3f}")
    print(f"   Final month AUC:        {trends['roc_auc'][-1]:.3f}")

    print("\n[DRIFT DETECTION]")
    print(f"   Drift triggers:         {drift_summary['triggered_periods']} / {drift_summary['total_periods']}")
    print(f"   First failure month:    {evaluator.identify_failure_month()}")

    print("\n[ROBUSTNESS SUMMARY]")
    print(f"   PSI warnings:           {robust_summary['psi_warnings']}")
    print(f"   KS test failures:       {robust_summary['ks_failures']}")
    print(f"   Calibration OK:         {robust_summary['calibration_ok']}")
    print(f"   Time to failure:        {robust_summary['months_to_failure']} months" if robust_summary['months_to_failure'] else "   Time to failure:        Never")
    print(f"   Degradation pattern:    {robust_summary['degradation_pattern']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Experiment completed successfully!")
    print(f"  Results saved to: {OUTPUTS_DIR}")
    print("=" * 70)

    summary = {
        'baseline': {'accuracy': baseline_metrics['accuracy'], 'roc_auc': baseline_metrics['roc_auc']},
        'final_month': {'accuracy': trends['accuracy'][-1], 'roc_auc': trends['roc_auc'][-1]},
        'performance_drop': baseline_metrics['accuracy'] - trends['accuracy'][-1],
        'drift_triggers': drift_summary['triggered_periods'],
        'robustness': robust_summary
    }
    
    with open(OUTPUTS_DIR / 'experiment_summary.json', 'w') as f:
        def convert(o):
            if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
            if isinstance(o, (np.integer, np.int64, np.int32)): return int(o)
            if isinstance(o, (np.bool_, bool)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, dict): return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, list): return [convert(v) for v in o]
            return o
        json.dump(convert(summary), f, indent=2)


if __name__ == "__main__":
    main()
