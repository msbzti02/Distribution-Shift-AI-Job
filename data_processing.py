import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import (
    RAW_DATA_FILES, DATE_COLUMN, TARGET_COLUMN,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    TRAIN_START, TRAIN_END, DEPLOY_START, DEPLOY_END
)

def load_and_prepare_data(filepaths: Optional[list] = None) -> pd.DataFrame:
    if filepaths is None:
        filepaths = RAW_DATA_FILES
    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        if 'salary_local' in df.columns:
            df = df.drop(columns=['salary_local'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['job_id'], keep='first')
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"Loaded {len(df)} records from {len(filepaths)} files")
    print(f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
    return df

class SalaryBandEncoder:
    def __init__(self, n_bands: int = 4):
        self.n_bands = n_bands
        self.thresholds = None
        self.band_labels = None

    def fit(self, salaries: pd.Series) -> 'SalaryBandEncoder':
        quantile_values = [i / self.n_bands for i in range(1, self.n_bands)]
        self.thresholds = salaries.quantile(quantile_values).values
        self.band_labels = {
            0: f"< ${int(self.thresholds[0]/1000)}K",
            1: f"${int(self.thresholds[0]/1000)}K - ${int(self.thresholds[1]/1000)}K",
            2: f"${int(self.thresholds[1]/1000)}K - ${int(self.thresholds[2]/1000)}K",
            3: f"> ${int(self.thresholds[2]/1000)}K"
        }
        print(f"Salary band thresholds: {self.thresholds}")
        return self
    
    def transform(self, salaries: pd.Series) -> pd.Series:
        return pd.cut(salaries, bins=[-np.inf] + list(self.thresholds) + [np.inf],
                     labels=list(range(self.n_bands))).astype(int)


class SkillEncoder:
    def __init__(self, min_frequency: int = 10):
        self.min_frequency = min_frequency
        self.skill_vocabulary = None
        self.skill_to_idx = None
    
    def _parse_skills(self, skills_str: str) -> List[str]:
        if pd.isna(skills_str):
            return []
        skills = [s.strip().strip('"').strip("'") for s in str(skills_str).split(',')]
        return [s for s in skills if s]
    
    def fit(self, skills_series: pd.Series) -> 'SkillEncoder':
        skill_counts = {}
        for skills_str in skills_series:
            for skill in self._parse_skills(skills_str):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        self.skill_vocabulary = sorted([s for s, c in skill_counts.items() if c >= self.min_frequency])
        self.skill_to_idx = {s: i for i, s in enumerate(self.skill_vocabulary)}
        print(f"Skill vocabulary size: {len(self.skill_vocabulary)}")
        return self
    
    def transform(self, skills_series: pd.Series) -> np.ndarray:
        result = np.zeros((len(skills_series), len(self.skill_vocabulary)), dtype=np.float32)
        for i, skills_str in enumerate(skills_series):
            for skill in self._parse_skills(skills_str):
                if skill in self.skill_to_idx:
                    result[i, self.skill_to_idx[skill]] = 1.0
        return result


class FeaturePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.skill_encoder = SkillEncoder()
        self.salary_encoder = SalaryBandEncoder()
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, salary_column: str = 'salary_usd') -> 'FeaturePreprocessor':
        self.salary_encoder.fit(df[salary_column])
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna('Unknown').astype(str))
                self.label_encoders[col] = le
        
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            self.scaler.fit(df[num_cols].fillna(0))
        
        if 'required_skills' in df.columns:
            self.skill_encoder.fit(df['required_skills'])
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, include_target: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        features, feature_names = [], []
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in self.label_encoders:
                values = df[col].fillna('Unknown').astype(str)
                le = self.label_encoders[col]
                encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in values])
                features.append(encoded.reshape(-1, 1))
                feature_names.append(col)
        
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            features.append(self.scaler.transform(df[num_cols].fillna(0)))
            feature_names.extend(num_cols)
        
        if 'required_skills' in df.columns:
            features.append(self.skill_encoder.transform(df['required_skills']))
            feature_names.extend([f"skill_{s}" for s in self.skill_encoder.skill_vocabulary])
        
        X = np.hstack(features)
        self.feature_columns = feature_names
        
        y = self.salary_encoder.transform(df['salary_usd']).values if include_target and 'salary_usd' in df.columns else None
        return X, y



def get_temporal_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, pd.DataFrame]]]:
 
    print("Creating temporal splits...")
    train_start = pd.to_datetime(TRAIN_START)
    train_end = pd.to_datetime(TRAIN_END)
    deploy_end = pd.to_datetime(DEPLOY_END)
    train_df = df[(df[DATE_COLUMN] >= train_start) & (df[DATE_COLUMN] < train_end)].copy()
    deploy_df = df[df[DATE_COLUMN] >= train_end].copy()
    print(f"Training: {train_start.date()} to {train_end.date()}, {len(train_df)} samples")
    print(f"Deployment: {len(deploy_df)} samples")

    monthly_windows = []
    current = train_end
    print("\nMonthly windows:")
    
    while current < deploy_end:
        next_month = (current.replace(day=1) + pd.DateOffset(months=1))
        window_end = min(next_month, deploy_end)
        
        mask = (deploy_df[DATE_COLUMN] >= current) & (deploy_df[DATE_COLUMN] < window_end)
        month_df = deploy_df[mask].copy()
        
        if len(month_df) > 0:
            label = current.strftime("%Y-%m")
            monthly_windows.append((label, month_df))
            print(f"  {label}: {len(month_df)} samples")
        
        current = next_month
    
    return train_df, monthly_windows


def validate_no_leakage(train_df: pd.DataFrame, monthly_windows: List) -> bool:
    train_max = train_df[DATE_COLUMN].max()
    for label, month_df in monthly_windows:
        if month_df[DATE_COLUMN].min() <= train_max:
            print(f"ERROR: Leakage in {label}")
            return False
    print("\n[OK] No temporal leakage")
    return True
