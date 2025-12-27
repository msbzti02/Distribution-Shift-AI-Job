import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Any, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
from datetime import datetime
from collections import defaultdict
from config import PSI_THRESHOLD, KS_PVALUE_THRESHOLD, JS_THRESHOLD, FEATURE_DRIFT_RATIO

def calculate_psi_numerical(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    _, bin_edges = np.histogram(reference, bins=n_bins)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    ref_prop = np.clip(ref_counts / len(reference), 1e-10, 1)
    cur_prop = np.clip(cur_counts / len(current), 1e-10, 1)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def calculate_psi_categorical(reference: np.ndarray, current: np.ndarray) -> float:
    categories = np.unique(np.concatenate([reference, current]))
    ref_counts = {c: 0 for c in categories}
    cur_counts = {c: 0 for c in categories}
    
    for v in reference: ref_counts[v] = ref_counts.get(v, 0) + 1
    for v in current: cur_counts[v] = cur_counts.get(v, 0) + 1
    
    eps = 1e-10
    psi = 0.0
    for c in categories:
        ref_p = (ref_counts[c] + eps) / (len(reference) + eps * len(categories))
        cur_p = (cur_counts[c] + eps) / (len(current) + eps * len(categories))
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
    return psi


def calculate_feature_psi(ref_features: np.ndarray, cur_features: np.ndarray, 
                          feature_names: List[str]) -> Dict[str, float]:
    return {name: calculate_psi_numerical(ref_features[:, i], cur_features[:, i]) 
            for i, name in enumerate(feature_names)}

def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    stat, p = stats.ks_2samp(reference, current)
    return stat, p

def calculate_feature_ks(ref_features: np.ndarray, cur_features: np.ndarray,
                         feature_names: List[str], indices: List[int]) -> Dict:
    return {feature_names[i]: {'statistic': ks_test(ref_features[:, i], cur_features[:, i])[0],
                               'p_value': ks_test(ref_features[:, i], cur_features[:, i])[1]}
            for i in indices if i < len(feature_names)}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p) / np.sum(p), np.asarray(q) / np.sum(q)
    return float(jensenshannon(p, q, base=2) ** 2)


def calculate_prediction_js(ref_probs: np.ndarray, cur_probs: np.ndarray) -> float:
    return js_divergence(ref_probs.mean(axis=0), cur_probs.mean(axis=0))


def parse_skills(skills_str: str) -> Set[str]:
    if not skills_str or str(skills_str) == 'nan':
        return set()
    return {s.strip().strip('"').strip("'") for s in str(skills_str).split(',') if s.strip()}


def get_skill_frequencies(skills_series) -> Dict[str, int]:

    freq = defaultdict(int)
    for skills_str in skills_series:
        for skill in parse_skills(skills_str):
            freq[skill] += 1
    return dict(freq)


def calculate_skill_churn(ref_skills: Dict[str, int], cur_skills: Dict[str, int]) -> Dict:
    ref_vocab, cur_vocab = set(ref_skills.keys()), set(cur_skills.keys())
    new_skills = cur_vocab - ref_vocab
    decayed = ref_vocab - cur_vocab
    overlap = ref_vocab & cur_vocab
    union = ref_vocab | cur_vocab
    
    return {
        'new_skills': list(new_skills), 'decayed_skills': list(decayed),
        'new_skill_count': len(new_skills), 'decayed_skill_count': len(decayed),
        'new_skill_rate': len(new_skills) / len(cur_vocab) if cur_vocab else 0,
        'decay_rate': len(decayed) / len(ref_vocab) if ref_vocab else 0,
        'vocabulary_overlap': len(overlap) / len(union) if union else 1.0,
    }


class SkillChurnTracker:
    def __init__(self):
        self.reference_skills = None
        self.history = []
    
    def set_reference(self, skills_series) -> None:
        self.reference_skills = get_skill_frequencies(skills_series)
        print(f"Reference vocabulary: {len(self.reference_skills)} unique skills")
    
    def update(self, skills_series, period_label: str) -> Dict:
        current = get_skill_frequencies(skills_series)
        churn = calculate_skill_churn(self.reference_skills, current)
        churn['period'] = period_label
        self.history.append(churn)
        return churn


class DriftDetector:
    def __init__(self, psi_threshold=PSI_THRESHOLD, ks_threshold=KS_PVALUE_THRESHOLD,
                 js_threshold=JS_THRESHOLD, feature_drift_ratio=FEATURE_DRIFT_RATIO):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        self.feature_drift_ratio = feature_drift_ratio
        self.reference_features = None
        self.reference_predictions = None
        self.feature_names = None
        self.skill_tracker = SkillChurnTracker()
        self.drift_history = []
        self.triggers = []
    
    def set_reference(self, features: np.ndarray, predictions: np.ndarray,
                      feature_names: List[str], skills_series=None) -> None:
        self.reference_features = features
        self.reference_predictions = predictions
        self.feature_names = feature_names
        if skills_series is not None:
            self.skill_tracker.set_reference(skills_series)
    
    def detect(self, features: np.ndarray, predictions: np.ndarray,
               period_label: str, skills_series=None) -> Dict:
        results = {'period': period_label, 'triggered': False, 'trigger_reasons': []}
        
        psi_values = calculate_feature_psi(self.reference_features, features, self.feature_names)
        results['psi'] = psi_values
        drifted = [n for n, p in psi_values.items() if p >= self.psi_threshold]
        results['drifted_features'] = drifted
        results['drift_ratio'] = len(drifted) / len(self.feature_names)
        
        if results['drift_ratio'] >= self.feature_drift_ratio:
            results['triggered'] = True
            results['trigger_reasons'].append(f"PSI drift ratio {results['drift_ratio']:.2%}")
        
        ks_results = calculate_feature_ks(self.reference_features, features, 
                                          self.feature_names, list(range(min(10, len(self.feature_names)))))
        results['ks_tests'] = ks_results
        results['ks_drifted_features'] = [n for n, r in ks_results.items() if r['p_value'] < self.ks_threshold]
        
        js_pred = calculate_prediction_js(self.reference_predictions, predictions)
        results['js_divergence'] = js_pred
        if js_pred >= self.js_threshold:
            results['triggered'] = True
            results['trigger_reasons'].append(f"JS divergence {js_pred:.4f}")
        
        if skills_series is not None:
            results['skill_churn'] = self.skill_tracker.update(skills_series, period_label)
        
        self.drift_history.append(results)
        if results['triggered']:
            self.triggers.append({'period': period_label, 'reasons': results['trigger_reasons']})
        
        return results
    
    def get_summary(self) -> Dict:
        return {'total_periods': len(self.drift_history), 'triggered_periods': len(self.triggers), 'triggers': self.triggers}
