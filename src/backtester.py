# src/backtester.py

import vectorbt as vbt
import pandas as pd
import numpy as np
import os

class VectorizedBacktester:
    """
    Performs a vectorized backtest using the vectorbt library.
    """
    def __init__(self, price_data, signals, config):
        """
        Initializes the backtester.

        Args:
            price_data (pd.DataFrame): DataFrame with at least 'open' and 'close' prices.
            signals (pd.Series): Series of binary signals (1 for entry, 0 for hold/exit).
            config (dict): The project configuration dictionary.
        """
        self.price_data = price_data
        self.signals = signals
        self.config = config

    def run(self, init_cash=100000, commission=0.001, slippage=0.001):
        """
        Runs the backtest and prints the results.

        Args:
            init_cash (float): The initial cash for the portfolio.
            commission (float): Transaction fee per trade.
            slippage (float): Slippage per trade.
        """
        print("\n===== Starting Vectorized Backtest with vectorbt =====")
        
        entries = self.signals == 1
        # Define exit signals for when the model's prediction is 0
        exits = self.signals == 0
        
        price = self.price_data['close']

        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            sl_stop=0.10,
            tp_stop=0.20,
            fees=commission,
            slippage=slippage,
            freq='D'
        )
        
        print("\n--- Backtest Performance Stats ---")
        print(pf.stats())

        print("\n--- Plotting Equity Curve and Drawdowns ---")
        
        # 1. Plot the main portfolio equity curve and drawdowns
        fig = pf.plot()

        # 2. Calculate the benchmark's equity curve
        benchmark_rets = price.pct_change()
        benchmark_equity = (1 + benchmark_rets).cumprod()
        
        # 3. Add the benchmark's equity curve to the SAME plot
        benchmark_equity.vbt.plot(fig=fig, trace_kwargs=dict(name='Benchmark', line=dict(color='gray')))
        
        # Now, show the combined figure
        fig.show()

        print("\n===== Backtest Finished =====")
        return pf

    def export_trades_to_csv(self, portfolio, filename, project_root):
        """
        Exports the closed trades from a portfolio object to a CSV file.

        Args:
            portfolio (vbt.Portfolio): The portfolio object from a backtest run.
            filename (str): The name of the output CSV file.
            project_root (str): The absolute path to the project's root directory.
        """
        # Create a 'results' directory if it doesn't exist
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Get the trade records as a pandas DataFrame
        trades_df = pd.DataFrame(portfolio.trades.records)
        
        # Define the full path for the output file
        output_path = os.path.join(results_dir, filename)
        
        if trades_df.empty:
            print(f"No trades to export for {filename}.")
        else:
            # Save the DataFrame to a CSV file
            trades_df.to_csv(output_path)
            print(f"Successfully exported {len(trades_df)} trades to: {output_path}")