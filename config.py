import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_DATA_FILES = [
    PROJECT_ROOT / "ai_job_dataset.csv",
    PROJECT_ROOT / "ai_job_dataset1.csv"
]
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-06-30"
DEPLOY_START = "2024-07-01"
DEPLOY_END = "2025-04-30"

EVAL_WINDOW_FREQ = "M"  

TARGET_COLUMN = "salary_band"
SALARY_BANDS = {
    0: "< $60K",
    1: "$60K - $100K",
    2: "$100K - $150K",
    3: "> $150K"
}
SALARY_THRESHOLDS = [60000, 100000, 150000] 

CATEGORICAL_FEATURES = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_location",
    "company_size",
    "industry",
    "education_required"
]
NUMERICAL_FEATURES = [
    "years_experience",
    "remote_ratio",
    "job_description_length",
    "benefits_score"
]
TEXT_FEATURES = ["required_skills"]
DATE_COLUMN = "posting_date"
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

PSI_THRESHOLD = 0.2 
KS_PVALUE_THRESHOLD = 0.05  
JS_THRESHOLD = 0.1  
FEATURE_DRIFT_RATIO = 0.3  

RANDOM_SEED = 42
