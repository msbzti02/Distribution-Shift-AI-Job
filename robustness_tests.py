import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from config import OUTPUTS_DIR, PSI_THRESHOLD, KS_PVALUE_THRESHOLD

@dataclass
class PSIResult:
    value: float
    status: str 
    @classmethod
    def from_value(cls, psi: float) -> 'PSIResult':
        if psi < 0.1:
            return cls(psi, 'stable')
        elif psi < 0.25:
            return cls(psi, 'warning')
        else:
            return cls(psi, 'high_risk')


class DataRobustnessTests:
    def __init__(self, reference_features: np.ndarray, feature_names: List[str]):
        self.reference = reference_features
        self.feature_names = feature_names
        self.reference_stats = self._compute_stats(reference_features)
    
    def _compute_stats(self, features: np.ndarray) -> Dict:
        return {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0),
            'p25': np.percentile(features, 25, axis=0),
            'p50': np.percentile(features, 50, axis=0),
            'p75': np.percentile(features, 75, axis=0)
        }
    
    def test_psi_per_feature(self, current: np.ndarray, n_bins: int = 10) -> Dict[str, PSIResult]:
        results = {}
        for i, name in enumerate(self.feature_names):
            ref_col, cur_col = self.reference[:, i], current[:, i]
            _, bin_edges = np.histogram(ref_col, bins=n_bins)
            ref_counts, _ = np.histogram(ref_col, bins=bin_edges)
            cur_counts, _ = np.histogram(cur_col, bins=bin_edges)
            ref_prop = np.clip(ref_counts / len(ref_col), 1e-10, 1)
            cur_prop = np.clip(cur_counts / len(cur_col), 1e-10, 1)
            psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
            results[name] = PSIResult.from_value(abs(psi))
        return results
    
    def test_ks_per_feature(self, current: np.ndarray) -> Dict[str, Dict]:
        results = {}
        for i, name in enumerate(self.feature_names[:10]): 
            stat, p_value = stats.ks_2samp(self.reference[:, i], current[:, i])
            results[name] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        return results
    
    def test_mean_variance_drift(self, current: np.ndarray) -> Dict[str, Dict]:
        current_stats = self._compute_stats(current)
        results = {}
        
        for i, name in enumerate(self.feature_names[:15]):
            ref_mean, cur_mean = self.reference_stats['mean'][i], current_stats['mean'][i]
            ref_std, cur_std = self.reference_stats['std'][i], current_stats['std'][i]
            
            mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
            std_ratio = cur_std / (ref_std + 1e-10)
            
            results[name] = {
                'ref_mean': float(ref_mean), 'cur_mean': float(cur_mean),
                'ref_std': float(ref_std), 'cur_std': float(cur_std),
                'mean_shift_zscore': float(mean_shift),
                'std_ratio': float(std_ratio),
                'drifted': mean_shift > 2 or std_ratio < 0.5 or std_ratio > 2
            }
        return results

class SkillDriftTests:
    
    def __init__(self, training_skills: pd.Series):
        self.training_vocab = self._extract_vocab(training_skills)
        self.training_freq = self._extract_freq(training_skills)
        self.top_skills = sorted(self.training_freq.keys(), 
                                  key=lambda x: self.training_freq[x], reverse=True)[:20]
    
    def _parse_skills(self, s: str) -> set:
        if pd.isna(s): return set()
        return {sk.strip().strip('"').strip("'") for sk in str(s).split(',') if sk.strip()}
    
    def _extract_vocab(self, series: pd.Series) -> set:
        vocab = set()
        for s in series:
            vocab.update(self._parse_skills(s))
        return vocab
    
    def _extract_freq(self, series: pd.Series) -> Dict[str, int]:
        freq = defaultdict(int)
        for s in series:
            for sk in self._parse_skills(s):
                freq[sk] += 1
        return dict(freq)
    
    def test_skill_vocabulary_churn(self, current_skills: pd.Series) -> Dict:
        current_vocab = self._extract_vocab(current_skills)
        
        unseen = current_vocab - self.training_vocab
        lost = self.training_vocab - current_vocab
        overlap = current_vocab & self.training_vocab
        
        return {
            'training_vocab_size': len(self.training_vocab),
            'current_vocab_size': len(current_vocab),
            'unseen_skills': len(unseen),
            'unseen_pct': len(unseen) / len(current_vocab) if current_vocab else 0,
            'lost_skills': len(lost),
            'overlap_pct': len(overlap) / len(self.training_vocab) if self.training_vocab else 0,
            'top_unseen': list(unseen)[:10]
        }
    
    def test_rare_skill_sensitivity(self, model, preprocessor, test_df: pd.DataFrame) -> Dict:
        X_full, y = preprocessor.transform(test_df)
        base_acc = (model.predict(X_full) == y).mean()

        skill_indices = [i for i, n in enumerate(preprocessor.feature_columns) if n.startswith('skill_')]
        
        if not skill_indices:
            return {'base_accuracy': float(base_acc), 'masked_accuracy': float(base_acc), 'drop': 0}
    
        X_masked = X_full.copy()
        top_skill_indices = skill_indices[:10] 
        X_masked[:, top_skill_indices] = 0
        masked_acc = (model.predict(X_masked) == y).mean()
        return {
            'base_accuracy': float(base_acc),
            'masked_accuracy': float(masked_acc),
            'accuracy_drop': float(base_acc - masked_acc),
            'brittle': (base_acc - masked_acc) > 0.1
        }


class LabelRobustnessTests:
    
    def __init__(self, training_labels: np.ndarray, n_classes: int = 4):
        self.n_classes = n_classes
        self.training_priors = np.bincount(training_labels, minlength=n_classes) / len(training_labels)
    
    def test_class_prior_drift(self, current_labels: np.ndarray) -> Dict:
        current_priors = np.bincount(current_labels, minlength=self.n_classes) / len(current_labels)
        
        prior_shift = np.abs(current_priors - self.training_priors)
        
        return {
            'training_priors': self.training_priors.tolist(),
            'current_priors': current_priors.tolist(),
            'prior_shift': prior_shift.tolist(),
            'max_shift': float(prior_shift.max()),
            'significant_shift': prior_shift.max() > 0.1
        }
    
    def test_expected_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                         n_bins: int = 10) -> Dict:
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece, mce = 0.0, 0.0
        bin_data = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                gap = abs(avg_acc - avg_conf)
                prop = in_bin.mean()
                ece += prop * gap
                mce = max(mce, gap)
                bin_data.append({'bin': f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}',
                                'confidence': float(avg_conf), 'accuracy': float(avg_acc), 'gap': float(gap)})
        
        class_ece = {}
        for c in range(self.n_classes):
            mask = y_true == c
            if mask.sum() > 0:
                class_conf = y_proba[mask, c]
                class_acc = (predictions[mask] == c)
                class_ece[f'class_{c}'] = float(abs(class_conf.mean() - class_acc.mean()))
        
        return {
            'ece': float(ece), 'mce': float(mce),
            'bins': bin_data, 'class_ece': class_ece,
            'calibration_ok': ece < 0.1
        }
    
    def test_confidence_collapse(self, y_proba: np.ndarray) -> Dict:
        max_probs = np.max(y_proba, axis=1)
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        
        return {
            'mean_confidence': float(max_probs.mean()),
            'std_confidence': float(max_probs.std()),
            'mean_entropy': float(entropy.mean()),
            'std_entropy': float(entropy.std()),
            'low_confidence_pct': float((max_probs < 0.5).mean()),
            'high_entropy_pct': float((entropy > 1.0).mean())
        }


class ConceptRobustnessTests:
    
    def test_skill_salary_mapping_stability(self, model, preprocessor, 
                                            train_df: pd.DataFrame, 
                                            test_df: pd.DataFrame) -> Dict:
        results = {}

        skill_groups = {
            'python_ml': ['Python', 'Machine Learning', 'TensorFlow', 'PyTorch'],
            'data_eng': ['SQL', 'Spark', 'ETL', 'Data Pipeline'],
            'cloud': ['AWS', 'Azure', 'GCP', 'Cloud']
        }
        
        for group_name, keywords in skill_groups.items():
            def has_skills(s):
                if pd.isna(s): return False
                return any(k.lower() in str(s).lower() for k in keywords)
            
            train_mask = train_df['required_skills'].apply(has_skills) if 'required_skills' in train_df.columns else pd.Series([False]*len(train_df))
            test_mask = test_df['required_skills'].apply(has_skills) if 'required_skills' in test_df.columns else pd.Series([False]*len(test_df))
            
            train_subset = train_df[train_mask]
            test_subset = test_df[test_mask]
            
            if len(train_subset) > 10 and len(test_subset) > 10:
                X_train_sub, y_train_sub = preprocessor.transform(train_subset)
                X_test_sub, y_test_sub = preprocessor.transform(test_subset)
                
                train_preds = model.predict(X_train_sub)
                test_preds = model.predict(X_test_sub)
                
                train_dist = np.bincount(train_preds, minlength=4) / len(train_preds)
                test_dist = np.bincount(test_preds, minlength=4) / len(test_preds)
                
                results[group_name] = {
                    'train_samples': len(train_subset),
                    'test_samples': len(test_subset),
                    'train_pred_dist': train_dist.tolist(),
                    'test_pred_dist': test_dist.tolist(),
                    'dist_shift': float(np.abs(train_dist - test_dist).sum()),
                    'concept_stable': np.abs(train_dist - test_dist).sum() < 0.3
                }
        
        return results
    
    def test_error_concentration(self, model, preprocessor, test_df: pd.DataFrame,
                                  group_column: str = 'experience_level') -> Dict:
        X, y = preprocessor.transform(test_df)
        preds = model.predict(X)
        errors = (preds != y)
        
        results = {}
        if group_column in test_df.columns:
            for group in test_df[group_column].unique():
                mask = test_df[group_column].values == group
                if mask.sum() > 0:
                    group_error_rate = errors[mask].mean()
                    results[str(group)] = {
                        'samples': int(mask.sum()),
                        'error_rate': float(group_error_rate),
                        'accuracy': float(1 - group_error_rate)
                    }
        
        if results:
            error_rates = [r['error_rate'] for r in results.values()]
            results['_summary'] = {
                'max_error_rate': float(max(error_rates)),
                'min_error_rate': float(min(error_rates)),
                'error_spread': float(max(error_rates) - min(error_rates)),
                'concentrated': (max(error_rates) - min(error_rates)) > 0.15
            }
        
        return results

class TemporalRobustnessTests:
    def __init__(self, baseline_metrics: Dict):
        self.baseline = baseline_metrics
    
    def test_time_to_failure(self, monthly_metrics: List[Dict], 
                              acc_threshold: float = 0.1,
                              ece_threshold: float = 0.15) -> Dict:
        baseline_acc = self.baseline['accuracy']
        
        failure_month = None
        months_to_failure = None
        
        for i, m in enumerate(monthly_metrics):
            acc_drop = baseline_acc - m['accuracy']
            ece = m['calibration']['ece']
            
            if acc_drop > acc_threshold or ece > ece_threshold:
                failure_month = m.get('month', f'month_{i}')
                months_to_failure = i + 1
                break
        
        return {
            'baseline_accuracy': float(baseline_acc),
            'failure_threshold': acc_threshold,
            'ece_threshold': ece_threshold,
            'failure_month': failure_month,
            'months_to_failure': months_to_failure,
            'survived_all': failure_month is None
        }
    
    def test_degradation_pattern(self, monthly_metrics: List[Dict]) -> Dict:
        accs = [m['accuracy'] for m in monthly_metrics]
        
        if len(accs) < 3:
            return {'pattern': 'insufficient_data'}

        changes = np.diff(accs)
        max_drop = abs(min(changes)) if len(changes) > 0 else 0
        gradual = max_drop < 0.05
        
        return {
            'monthly_accuracies': accs,
            'month_changes': changes.tolist(),
            'max_single_drop': float(max_drop),
            'pattern': 'gradual' if gradual else 'sudden',
            'stable': all(abs(c) < 0.03 for c in changes)
        }

class SystemRobustnessTests:
    
    def test_drift_detection_accuracy(self, drift_triggers: List[Dict], 
                                       performance_drops: List[bool]) -> Dict:
        if len(drift_triggers) != len(performance_drops):
            return {'error': 'Length mismatch'}
        tp, fp, tn, fn = 0, 0, 0, 0
        for trigger, dropped in zip(drift_triggers, performance_drops):
            triggered = trigger.get('triggered', False)
            if triggered and dropped:
                tp += 1
            elif triggered and not dropped:
                fp += 1
            elif not triggered and dropped:
                fn += 1
            else:
                tn += 1
        
        total = tp + fp + tn + fn
        
        return {
            'true_positives': tp, 'false_positives': fp,
            'true_negatives': tn, 'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fp_rate': fp / total if total > 0 else 0,
            'fn_rate': fn / total if total > 0 else 0
        }
    
    def test_feature_masking(self, model, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Dict:
        base_acc = (model.predict(X) == y).mean()
        results = {'base_accuracy': float(base_acc)}
    
        groups = {
            'categorical': [i for i, n in enumerate(feature_names) if not n.startswith('skill_')][:5],
            'skills': [i for i, n in enumerate(feature_names) if n.startswith('skill_')][:20],
            'numerical': list(range(5, 10))
        }
        
        for group_name, indices in groups.items():
            if indices:
                X_masked = X.copy()
                X_masked[:, indices] = 0
                masked_acc = (model.predict(X_masked) == y).mean()
                results[f'{group_name}_masked'] = {
                    'accuracy': float(masked_acc),
                    'drop': float(base_acc - masked_acc),
                    'robust': (base_acc - masked_acc) < 0.1
                }
        
        return results


def run_all_robustness_tests(
    model, preprocessor, baseline_metrics: Dict,
    train_df: pd.DataFrame, monthly_windows: List[Tuple[str, pd.DataFrame]],
    drift_history: List[Dict], monthly_metrics: List[Dict]
) -> Dict[str, Any]:
    
    print("\n" + "="*60)
    print("ROBUSTNESS TESTING SUITE")
    print("="*60)
    
    results = {}

    X_train, y_train = preprocessor.transform(train_df)
    
    # Combine deployment data for testing
    deploy_dfs = [df for _, df in monthly_windows]
    if deploy_dfs:
        deploy_df = pd.concat(deploy_dfs, ignore_index=True)
        X_deploy, y_deploy = preprocessor.transform(deploy_df)
    else:
        return {'error': 'No deployment data'}
    
    print("\n[1/5] Data Robustness Tests...")
    data_tests = DataRobustnessTests(X_train, preprocessor.feature_columns)
    results['data_robustness'] = {
        'psi': {k: {'value': v.value, 'status': v.status} 
                for k, v in data_tests.test_psi_per_feature(X_deploy).items()},
        'ks_tests': data_tests.test_ks_per_feature(X_deploy),
        'mean_variance_drift': data_tests.test_mean_variance_drift(X_deploy)
    }
    
    if 'required_skills' in train_df.columns:
        skill_tests = SkillDriftTests(train_df['required_skills'])
        results['skill_drift'] = {
            'vocabulary_churn': skill_tests.test_skill_vocabulary_churn(deploy_df['required_skills']),
            'rare_skill_sensitivity': skill_tests.test_rare_skill_sensitivity(model, preprocessor, deploy_df)
        }
    
    print("[2/5] Label Robustness Tests...")
    label_tests = LabelRobustnessTests(y_train)
    y_proba_deploy = model.predict_proba(X_deploy)
    results['label_robustness'] = {
        'class_prior_drift': label_tests.test_class_prior_drift(y_deploy),
        'calibration_error': label_tests.test_expected_calibration_error(y_deploy, y_proba_deploy),
        'confidence_collapse': label_tests.test_confidence_collapse(y_proba_deploy)
    }
    
    print("[3/5] Concept Robustness Tests...")
    concept_tests = ConceptRobustnessTests()
    results['concept_robustness'] = {
        'skill_salary_mapping': concept_tests.test_skill_salary_mapping_stability(
            model, preprocessor, train_df, deploy_df),
        'error_concentration': concept_tests.test_error_concentration(
            model, preprocessor, deploy_df, 'experience_level')
    }
    
    print("[4/5] Temporal Robustness Tests...")
    temporal_tests = TemporalRobustnessTests(baseline_metrics)
    results['temporal_robustness'] = {
        'time_to_failure': temporal_tests.test_time_to_failure(monthly_metrics),
        'degradation_pattern': temporal_tests.test_degradation_pattern(monthly_metrics)
    }

    print("[5/5] System-Level Tests...")
    system_tests = SystemRobustnessTests()

    performance_drops = [m['accuracy'] < baseline_metrics['accuracy'] - 0.05 for m in monthly_metrics]
    
    results['system_robustness'] = {
        'drift_detection_accuracy': system_tests.test_drift_detection_accuracy(drift_history, performance_drops),
        'feature_masking': system_tests.test_feature_masking(model, X_deploy, y_deploy, preprocessor.feature_columns)
    }
    
    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    
    psi_warnings = sum(1 for v in results['data_robustness']['psi'].values() if v['status'] != 'stable')
    ks_failures = sum(1 for v in results['data_robustness']['ks_tests'].values() if v.get('significant', False))
    calibration_ok = results['label_robustness']['calibration_error']['calibration_ok']
    ttf = results['temporal_robustness']['time_to_failure']
    
    summary = {
        'psi_warnings': psi_warnings,
        'ks_failures': ks_failures,
        'calibration_ok': calibration_ok,
        'months_to_failure': ttf['months_to_failure'],
        'degradation_pattern': results['temporal_robustness']['degradation_pattern']['pattern']
    }
    
    print(f"  PSI warnings: {psi_warnings}")
    print(f"  KS test failures: {ks_failures}")
    print(f"  Calibration OK: {calibration_ok}")
    print(f"  Time to failure: {ttf['months_to_failure']} months" if ttf['months_to_failure'] else "  Time to failure: Did not fail")
    print(f"  Degradation: {results['temporal_robustness']['degradation_pattern']['pattern']}")
    
    results['summary'] = summary
    
    return results


def save_robustness_results(results: Dict, filepath: Optional[Path] = None):
    if filepath is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = OUTPUTS_DIR / 'robustness_results.json'
    summary = results.get('summary', {})

    save_data = {
        'psi_warnings': int(summary.get('psi_warnings', 0)),
        'ks_failures': int(summary.get('ks_failures', 0)),
        'calibration_ok': bool(summary.get('calibration_ok', True)),
        'months_to_failure': summary.get('months_to_failure'),
        'degradation_pattern': str(summary.get('degradation_pattern', 'unknown'))
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n[OK] Robustness summary saved to {filepath}")
    return filepath
