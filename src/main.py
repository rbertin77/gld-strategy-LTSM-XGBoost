# Em src/main.py

import yaml
from src.training_pipeline import TrainingPipeline
from src.utils import set_seeds  # <--- 1. Importe a nova função

def main():
    """
    Main function to run the ML pipeline.
    """
    print("--- Starting Project: Deep Learning for Asset Prediction ---")
    
    # Load configuration
    print("Loading configuration from config.yaml...")
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # --- 2. Set seeds for reproducibility ---
    set_seeds(config['SEED'])
    
    # Instantiate and run the training pipeline
    pipeline = TrainingPipeline(config=config)
    pipeline.run()
    
    print("--- Project Finished ---")

if __name__ == "__main__":
    main()
    