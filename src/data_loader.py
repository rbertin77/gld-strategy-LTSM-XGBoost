# src/data_loader.py

import os
import pandas as pd
import yfinance as yf

class DataLoader:
    """
    Handles all data loading operations, including downloading from Yahoo Finance
    and caching to local CSV files.
    """
    def __init__(self, config, project_root):
        self.config = config
        self.data_dir = os.path.join(project_root, self.config['DATA_DIR'])
        os.makedirs(self.data_dir, exist_ok=True)

    
    def _download_and_cache(self, ticker, start_date, end_date):
        """
        A private helper method to download data for a single ticker and cache it.
        """
        file_path = os.path.join(self.data_dir, f"{ticker.lower()}_data.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found for {ticker}. Downloading data...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            # Clean potential MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.lower() for col in df.columns]
            df.to_csv(file_path)
            print(f"Data for {ticker} saved to {file_path}")
        else:
            print(f"Loading {ticker} data from local cache: {file_path}")
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df

    def load_main_data(self):
        """
        Loads the main ticker data for the project (e.g., GLD).
        """
        print("--- Loading Main Asset Data ---")
        return self._download_and_cache(
            self.config['TICKER'],
            self.config['START_DATE'],
            self.config['END_DATE']
        )
        
    def load_all_macro_data(self):
        """
        Loads all macroeconomic ticker data defined in the config.
        """
        print("\n--- Loading Macroeconomic Data ---")
        macro_data = {}
        for feature_name, details in self.config['MACRO_TICKERS'].items():
            ticker = details['ticker']
            df_feature = self._download_and_cache(
                ticker,
                self.config['START_DATE'],
                self.config['END_DATE']
            )
            
            if not df_feature.empty:
                # Select only the 'Close' price and rename it
                macro_data[details['name']] = df_feature['close'].rename(details['name'])
            else:
                print(f" -> WARNING: No data downloaded for ticker '{ticker}'. Skipping this feature.")
        
        # Combine all series into a single dataframe
        return pd.concat(macro_data.values(), axis=1)