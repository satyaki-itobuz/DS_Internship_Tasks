import os
from src.merging_stratification import (
    load_datasets,
    merge_datasets,
    treat_missing_values,
    stratified_sample,
    save_to_csv,
)
from src.data_encoding import preprocess_data, csv_save

if __name__ == "__main__":
    # Define folder paths relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.abspath(os.path.join(base_dir, "data"))
    sampled_file = os.path.join(data_folder, "sampled_data.csv")
    processed_file = os.path.join(data_folder, "processed_data.csv")

    datasets = load_datasets(data_folder)
    if datasets:
        merged_df = merge_datasets(datasets)

        if merged_df is not None:
            treated_df = treat_missing_values(merged_df)

            if treated_df is not None:
                sampled_df = stratified_sample(treated_df, stratify_col="reordered", frac=0.3)

                if not sampled_df.empty:
                    save_to_csv(sampled_df, data_folder, "sampled_data.csv")

                    preprocess_data(sampled_file, processed_file)
                else:
                    print("Error: Stratified sampling resulted in an empty DataFrame.")
            else:
                print("Error: Treating missing values failed.")
        else:
            print("Error: Merging datasets failed.")
    else:
        print("Error: Loading datasets failed.")