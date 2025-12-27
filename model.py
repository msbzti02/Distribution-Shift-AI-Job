import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import pickle
import json
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from config import XGBOOST_PARAMS, MODELS_DIR


class BaselineModel:
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or XGBOOST_PARAMS
        self.model = XGBClassifier(**self.params, objective='multi:softprob', num_class=4,
                                   use_label_encoder=False, eval_metric='mlogloss')
        self.training_metrics = {}
        self.feature_distributions = {}
        self.class_priors = {}
        self.training_date = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> 'BaselineModel':
        self.training_date = datetime.now().isoformat()
        self.model.fit(X, y)
        
        unique, counts = np.unique(y, return_counts=True)
        self.class_priors = {int(k): int(v) for k, v in zip(unique, counts)}
        self.feature_distributions = {
            'mean': X.mean(axis=0).tolist(), 'std': X.std(axis=0).tolist(),
            'min': X.min(axis=0).tolist(), 'max': X.max(axis=0).tolist()
        }
        
        y_pred, y_proba = self.model.predict(X), self.model.predict_proba(X)
        self.training_metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'roc_auc': float(roc_auc_score(y, y_proba, multi_class='ovr')),
            'n_samples': len(y), 'n_features': X.shape[1]
        }
        print(f"Training accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"Training ROC-AUC: {self.training_metrics['roc_auc']:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / 'baseline_model.pkl'
        
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'training_metrics': self.training_metrics,
                        'feature_distributions': self.feature_distributions,
                        'class_priors': self.class_priors, 'training_date': self.training_date,
                        'params': self.params}, f)
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BaselineModel':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(params=data['params'])
        instance.model = data['model']
        instance.training_metrics = data['training_metrics']
        instance.feature_distributions = data['feature_distributions']
        instance.class_priors = data['class_priors']
        instance.training_date = data['training_date']
        return instance


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, 
                   feature_names=None, save_model: bool = True) -> BaselineModel:
    print("Training baseline model...")
    model = BaselineModel()
    model.fit(X_train, y_train, feature_names)
    if save_model:
        model.save()
    return model
