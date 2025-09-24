# src/training_pipeline.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tabulate import tabulate
from sklearn.metrics import classification_report, roc_auc_score

# Importing our own modules
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import build_lstm_model
from src.xgboost_model import build_xgboost_model
from src.utils import set_seeds, plot_confusion_matrix, plot_roc_curve

# Helper function to reshape data for LSTM
def create_lstm_dataset(X, y, time_steps=1):
    """Reshapes 2D data to 3D for LSTMs."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

class TrainingPipeline:
    def __init__(self, config, project_root):
        self.config = config
        self.project_root = project_root
        set_seeds(self.config['SEED'])
        self.loader = DataLoader(config, self.project_root)
        self.feature_engineer = FeatureEngineer(config)

    def _load_and_prepare_data(self):
        """
        Private method to run the initial data loading and feature engineering steps once.
        """
        print("===== Step 1: Loading and Preparing Full Dataset =====")
        main_data = self.loader.load_main_data()
        macro_data = self.loader.load_all_macro_data()
        processed_data = self.feature_engineer.run(main_data, macro_data)
        return processed_data

    # --- MÉTODO PARA O TESTE ESTÁTICO ---
    def run_static_test(self, model_type='lstm'):
        """
        Runs a single train-validation-test split and backtest.
        Can train and evaluate either an 'lstm' or 'xgboost' model.
        
        Args:
            model_type (str): The type of model to run. Either 'lstm' or 'xgboost'.
        """
        print(f"\n\n===== Starting STATIC Test Run for model_type='{model_type}' =====")
        processed_data = self._load_and_prepare_data()
        
        X = processed_data.drop(columns=[self.config['TARGET_NAME']])
        y = processed_data[self.config['TARGET_NAME']]
        
        print("\n--- Splitting data chronologically ---")
        train_val_size = int(len(X) * (1 - self.config['TEST_SIZE']))
        X_train_val, X_test = X.iloc[:train_val_size], X.iloc[train_val_size:]
        y_train_val, y_test = y.iloc[:train_val_size], y.iloc[train_val_size:]
        val_split_index = int(len(X_train_val) * (1 - self.config['VALIDATION_SIZE']))
        X_train, X_val = X_train_val.iloc[:val_split_index], X_train_val.iloc[val_split_index:]
        y_train, y_val = y_train_val.iloc[:val_split_index], y_train_val.iloc[val_split_index:]
        print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")
        
        print("\n--- Running Feature Selection ---")
        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_train_filtered = X_train.drop(columns=to_drop)
        
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, random_state=self.config['SEED'])
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=self.config['SEED'])
        boruta_selector.fit(X_train_filtered.values, y_train.values)
        
        selected_features = X_train_filtered.columns[boruta_selector.support_].tolist()
        print(f"Selected {len(selected_features)} features via BorutaPy.")
        
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        X_test_selected = X_test[selected_features]
        
        print("\n--- Scaling data ---")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # --- MODEL-SPECIFIC LOGIC ---
        if model_type == 'lstm':
            print("\n--- Preparing data and training LSTM Model ---")
            time_steps = self.config['TIME_STEPS']
            
            # LSTM requires 3D data, so we reshape it
            X_train_final, y_train_final = create_lstm_dataset(X_train_scaled, y_train.values, time_steps)
            X_val_final, y_val_final = create_lstm_dataset(X_val_scaled, y_val.values, time_steps)
            X_test_final, y_test_final = create_lstm_dataset(X_test_scaled, y_test.values, time_steps)
            
            model = build_lstm_model(
                (X_train_final.shape[1], X_train_final.shape[2]),
                lstm_units=self.config['LSTM_UNITS'],
                dropout_rate=self.config['DROPOUT_RATE'],
                learning_rate=self.config['LEARNING_RATE']
            )
            
            history = model.fit(
                X_train_final, y_train_final, 
                epochs=100, batch_size=32, 
                validation_data=(X_val_final, y_val_final),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], 
                verbose=1
            )
            
            y_pred_proba_test = model.predict(X_test_final).ravel()

        elif model_type == 'xgboost':
            print("\n--- Preparing data and training XGBoost Model ---")
            # XGBoost uses 2D data directly, no reshaping needed
            X_train_final, y_train_final = X_train_scaled, y_train
            X_val_final, y_val_final = X_val_scaled, y_val
            X_test_final, y_test_final = X_test_scaled, y_test
            
            model = build_xgboost_model(random_seed=self.config['SEED'])
            
            # XGBoost uses Early Stopping via fit parameters
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                verbose=False
            )
            
            history = None # XGBoost doesn't have a Keras-like history object
            y_pred_proba_test = model.predict_proba(X_test_final)[:, 1]
        
        else:
            raise ValueError("Unsupported model_type. Choose 'lstm' or 'xgboost'.")

        print("\n--- Evaluating Model on Unseen Test Data ---")
        
        results = {
            "model": model,
            "history": history,
            "selected_features": selected_features,
            "scaler": scaler,
            "true_labels": y_test_final,
            "pred_probas": y_pred_proba_test,
            "processed_data_for_backtest": processed_data
        }
        return results

    # --- MÉTODO PARA WALK-FORWARD ---
    def run_walk_forward(self, model_type='lstm', n_splits=5):
        """
        Executes the entire walk-forward validation process for a given model type.
        
        Args:
            model_type (str): The type of model to run. Either 'lstm' or 'xgboost'.
            n_splits (int): The number of folds for the TimeSeriesSplit.
        """
        print(f"\n\n===== Starting WALK-FORWARD VALIDATION for model_type='{model_type}' =====")
        full_data = self._load_and_prepare_data()
        X = full_data.drop(columns=[self.config['TARGET_NAME']])
        y = full_data[self.config['TARGET_NAME']]

        tscv = TimeSeriesSplit(n_splits=n_splits)

        all_true_labels = []
        all_pred_probas = []
        all_backtest_data = []

        fold = 0
        for train_index, test_index in tscv.split(X):
            fold += 1
            print(f"\n{'='*20} FOLD {fold}/{n_splits} {'='*20}")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

            print("--- Running Feature Selection for this fold ---")
            corr_matrix = X_train.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            X_train_filtered = X_train.drop(columns=to_drop)
            
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, random_state=self.config['SEED'])
            boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=self.config['SEED'])
            boruta_selector.fit(X_train_filtered.values, y_train.values)
            
            selected_features = X_train_filtered.columns[boruta_selector.support_].tolist()
            print(f"Selected {len(selected_features)} features for fold {fold}.")
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # --- MODEL-SPECIFIC LOGIC WITHIN THE LOOP ---
            if model_type == 'lstm':
                print("--- Building and Training LSTM Model for this fold ---")
                time_steps = self.config['TIME_STEPS']
                
                X_train_final, y_train_final = create_lstm_dataset(X_train_scaled, y_train.values, time_steps)
                # For walk-forward, the test set of the fold serves as the validation set for EarlyStopping
                X_test_final, y_test_final = create_lstm_dataset(X_test_scaled, y_test.values, time_steps)
                
                if len(X_test_final) == 0:
                    print("Skipping fold due to insufficient data for test sequences.")
                    continue

                input_shape = (X_train_final.shape[1], X_train_final.shape[2])
                model = build_lstm_model(
                input_shape,
                lstm_units=self.config['LSTM_UNITS'],
                dropout_rate=self.config['DROPOUT_RATE'],
                learning_rate=self.config['LEARNING_RATE']
                )
                
                model.fit(
                    X_train_final, y_train_final,
                    epochs=100, batch_size=32,
                    validation_data=(X_test_final, y_test_final),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
                    verbose=0
                )
                
                y_pred_proba_test = model.predict(X_test_final).ravel()

            elif model_type == 'xgboost':
                print("--- Building and Training XGBoost Model for this fold ---")
                X_train_final, y_train_final = X_train_scaled, y_train
                X_test_final, y_test_final = X_test_scaled, y_test
                
                model = build_xgboost_model(random_seed=self.config['SEED'])
                
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_test_final, y_test_final)],
                    verbose=False
                )
                
                y_pred_proba_test = model.predict_proba(X_test_final)[:, 1]

            else:
                raise ValueError("Unsupported model_type. Choose 'lstm' or 'xgboost'.")

            # --- Store Out-of-Sample Predictions from the fold ---
            all_true_labels.extend(y_test_final)
            all_pred_probas.extend(y_pred_proba_test)
            
            # --- Store data needed for the final backtest ---
            # Correctly handle index alignment for LSTM vs XGBoost
            if model_type == 'lstm':
                time_steps = self.config['TIME_STEPS']
                fold_backtest_data = X_test.iloc[time_steps:].copy()
            else: # XGBoost
                fold_backtest_data = X_test.copy()

            fold_backtest_data['signal'] = (y_pred_proba_test > 0.5).astype(int)
            all_backtest_data.append(fold_backtest_data)

        print(f"\n{'='*20} Walk-Forward Validation Finished {'='*20}")
        
        final_backtest_df = pd.concat(all_backtest_data)
        
        results = {
            "true_labels": np.array(all_true_labels),
            "pred_probas": np.array(all_pred_probas),
            "backtest_df": final_backtest_df
        }
        return results