import logging
import pandas as pd
from src.feature_engineering import feature_engineering

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run feature engineering.
    """
    try:
        logger.info("Starting feature.py script.")
        
        # Load the dataset
        logger.info("Loading the dataset from 'data/product_dataset.csv'.")
        df = pd.read_csv('data/product_dataset.csv')
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Run the feature engineering process
        logger.info("Running feature engineering function from src.")
        processed_df = feature_engineering(df)
        logger.info(f"Feature engineering completed. Processed DataFrame shape: {processed_df.shape}")

        # Save the processed DataFrame
        output_file = 'data/processed_dataset.csv'
        logger.info(f"Saving the processed DataFrame to {output_file}.")
        processed_df.to_csv(output_file, index=False)
        logger.info("Processed DataFrame saved successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
