# src/xgboost_model.py

import xgboost as xgb

def build_xgboost_model(random_seed=42):
    """
    Builds an XGBoost classifier with a set of robust default parameters.

    Args:
        random_seed (int): Seed for reproducibility.

    Returns:
        An unfitted XGBoost Classifier model.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000,          # High number of estimators, will be controlled by early stopping
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=15,
        use_label_encoder=False,    # Recommended to avoid a deprecation warning
        random_state=random_seed
    )
    return model