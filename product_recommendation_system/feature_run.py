import logging
import pandas as pd
import os
from src.feature_engineering import feature_engineering

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/Users/itobuz/Desktop/project/DS_Internship_Tasks/product_recommendation_system/log/feature_engineering.log')
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
        dataset_path = '/Users/itobuz/Desktop/project/DS_Internship_Tasks/product_recommendation_system/data/product_dataset.csv'
        logger.info(f"Loading the dataset from '{dataset_path}'.")
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Run the feature engineering process
        logger.info("Running feature engineering function from src.")
        processed_df = feature_engineering(df)
        logger.info(f"Feature engineering completed. Processed DataFrame shape: {processed_df.shape}")

        # Save the processed DataFrame
        output_dir = '/Users/itobuz/Desktop/project/DS_Internship_Tasks/product_recommendation_system/data'
        output_file = os.path.join(output_dir, 'processed_dataset.csv')
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            logger.info(f"Output directory '{output_dir}' does not exist. Creating it.")
            os.makedirs(output_dir)

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
