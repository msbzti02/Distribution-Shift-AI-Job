import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from config import OUTPUTS_DIR


def calculate_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict:
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.mean() > 0:
            gap = abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
            mce = max(mce, gap)
    
    return {'ece': ece, 'mce': mce}

def calculate_prediction_entropy(y_proba: np.ndarray) -> Dict:
    proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    return {'mean_entropy': float(np.mean(entropy)), 'std_entropy': float(np.std(entropy))}


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except ValueError:
        auc = np.nan
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': auc,
        'calibration': calculate_calibration_error(y_true, y_proba),
        'entropy': calculate_prediction_entropy(y_proba)
    }


class MonthlyEvaluator:
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['< Q25', 'Q25-Q50', 'Q50-Q75', '> Q75']
        self.monthly_results = []
        self.baseline_metrics = None
    
    def set_baseline(self, metrics: Dict) -> None:
        self.baseline_metrics = metrics
        print(f"Baseline accuracy: {metrics['accuracy']:.4f}")
    
    def evaluate_month(self, model, X: np.ndarray, y: np.ndarray, 
                       month_label: str, role_labels=None) -> Dict:
        y_pred, y_proba = model.predict(X), model.predict_proba(X)
        metrics = calculate_all_metrics(y, y_pred, y_proba)
        metrics['month'] = month_label
        metrics['n_samples'] = len(y)
        
        if self.baseline_metrics:
            metrics['accuracy_delta'] = metrics['accuracy'] - self.baseline_metrics['accuracy']
        
        self.monthly_results.append(metrics)
        return metrics
    
    def get_trends(self) -> Dict[str, List]:
        return {
            'months': [r['month'] for r in self.monthly_results],
            'accuracy': [r['accuracy'] for r in self.monthly_results],
            'roc_auc': [r['roc_auc'] for r in self.monthly_results],
            'ece': [r['calibration']['ece'] for r in self.monthly_results],
            'mean_entropy': [r['entropy']['mean_entropy'] for r in self.monthly_results],
            'n_samples': [r['n_samples'] for r in self.monthly_results]
        }
    
    def identify_failure_month(self, threshold_drop: float = 0.05) -> Optional[str]:
        if not self.baseline_metrics:
            return None
        baseline_acc = self.baseline_metrics['accuracy']
        for r in self.monthly_results:
            if r['accuracy'] < baseline_acc - threshold_drop:
                return r['month']
        return None


class RollingWindowRetrainer:
    
    def __init__(self, window_months: int = 6, min_samples: int = 500):
        self.window_months = window_months
        self.min_samples = min_samples
        self.retraining_history = []
    
    def retrain(self, historical_data: List[Tuple[str, pd.DataFrame]], 
                preprocessor, trigger_month: str):
        from model import BaselineModel
        
        window_data = [df for _, df in historical_data[-self.window_months:]]
        combined = pd.concat(window_data, ignore_index=True)
        X, y = preprocessor.transform(combined)
        
        model = BaselineModel()
        model.fit(X, y)
        
        info = {
            'trigger_month': trigger_month,
            'n_samples': len(combined),
            'training_accuracy': model.training_metrics['accuracy']
        }
        self.retraining_history.append(info)
        print(f"Retrained on {len(combined)} samples")
        return model, info
    
    def get_retraining_cost(self) -> Dict:
        if not self.retraining_history:
            return {'total_retrains': 0, 'total_samples_processed': 0}
        return {
            'total_retrains': len(self.retraining_history),
            'total_samples_processed': sum(r['n_samples'] for r in self.retraining_history)
        }


class StrategyComparison:
    
    def __init__(self):
        self.static_results, self.adaptive_results, self.months = [], [], []
    
    def add_result(self, month: str, static: Dict, adaptive: Dict):
        self.months.append(month)
        self.static_results.append(static)
        self.adaptive_results.append(adaptive)
    
    def get_comparison(self) -> Dict:
        if not self.months:
            return {}
        static_acc = [r['accuracy'] for r in self.static_results]
        adaptive_acc = [r['accuracy'] for r in self.adaptive_results]
        return {
            'months': self.months,
            'static_accuracy': {'mean': np.mean(static_acc), 'values': static_acc},
            'adaptive_accuracy': {'mean': np.mean(adaptive_acc), 'values': adaptive_acc},
            'accuracy_improvement': {'mean': np.mean(np.array(adaptive_acc) - np.array(static_acc))}
        }
