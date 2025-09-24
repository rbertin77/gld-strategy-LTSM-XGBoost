# src/feature_engineering.py

import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    """
    Takes raw data and applies all feature engineering steps.
    """
    def __init__(self, config):
        self.config = config

    def _create_ohlcv_features(self, df):
        """Creates custom features from OHLCV data."""
        print("Step 1: Creating custom OHLCV features...")
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['close_change'] = (df['close'] - df['open']) / df['open']
        return df
        
    def _create_technical_indicators(self, df):
        """Applies the 'All' strategy from pandas_ta."""
        print("Step 2: Applying 'All' technical indicator strategy from pandas_ta...")
        df.ta.strategy('All', append=True, exclude=["mcgd"])

        # Clean up redundant columns
        cols_to_drop = [
            'HILO1_13_21', 'HILOS_13_21', 'PSAR1_0.02_0.2', 'PSARS_0.02_0.2',
            'QQE1_14_5_4.236', 'QQEs_14_5_4.236', 'SUPERT1_7_3.0', 'SUPERTS_7_3.0',
            'PSARaf_0.02_0.2', 'VTXt_14'
        ]
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df.drop(columns=existing_cols_to_drop, inplace=True)
        print(f" -> Dropped {len(existing_cols_to_drop)} redundant TA columns.")
        return df

    def _create_custom_interaction_features(self, df):
        """Creates custom features by combining existing ones."""
        print("Step 3: Creating custom interaction and ratio features...")
        # Ensure required base indicators exist before creating interactions
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.bbands(length=20, append=True)
        
        df['price_vs_ema50'] = df['close'] / df['EMA_50']
        df['ema_short_vs_long'] = df['EMA_10'] / df['EMA_50']
        df['price_vs_upper_band'] = df['close'] / df['BBU_20_2.0']
        df['price_vs_lower_band'] = df['close'] / df['BBL_20_2.0']
        return df
        
    def _create_lag_and_momentum_features(self, df):
        """Creates lagged returns and momentum of indicators."""
        print("Step 4: Creating lagged and momentum features...")
        df['returns'] = df['close'].pct_change() # Calculate returns first
        for i in range(1, 4):
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        df['volume_change_lag_1'] = df['volume'].pct_change().shift(1)
        
        if 'RSI_14' in df.columns:
            df['rsi_14_momentum_5d'] = df['RSI_14'].diff(5)
        if 'MFI_14' in df.columns:
            df['mfi_14_momentum_5d'] = df['MFI_14'].diff(5)
        
        return df

    def _merge_macro_data(self, main_df, macro_df):
        """Merges macroeconomic data into the main dataframe."""
        print("Step 5: Merging macroeconomic features...")
        df_augmented = pd.merge(main_df, macro_df, on='Date', how='left')
        df_augmented.ffill(inplace=True) # Use ffill to prevent lookahead bias
        print(" -> Macro features merged and forward-filled.")
        return df_augmented
        
    def _define_target(self, df):
        """Defines the target variable for the classification task."""
        print("Step 6: Defining target variable...")
        target_name = self.config['TARGET_NAME']
        df[target_name] = (df['close'].shift(-5) > df['close']).astype(int)
        return df
        
    def run(self, main_data, macro_data):
        """
        Executes the full feature engineering pipeline.
        
        Args:
            main_data (pd.DataFrame): The raw data for the main asset.
            macro_data (pd.DataFrame): The dataframe with all macro features.
            
        Returns:
            pd.DataFrame: The final dataframe with all features and the target.
        """
        print("\n===== Starting Feature Engineering Pipeline =====")
        df = main_data.copy()
        
        df = self._create_ohlcv_features(df)
        df = self._create_technical_indicators(df)
        df = self._create_custom_interaction_features(df)
        df = self._create_lag_and_momentum_features(df) # returns are created here
        df = self._merge_macro_data(df, macro_data)
        df = self._define_target(df)
        
        # Final cleanup
        initial_rows = df.shape[0]
        df.dropna(inplace=True)
        final_rows = df.shape[0]
        print(f"\nPipeline complete. Dropped {initial_rows - final_rows} rows with NaN values.")
        
        print(f"Final dataset shape: {df.shape}")
        print("===============================================\n")
        return df