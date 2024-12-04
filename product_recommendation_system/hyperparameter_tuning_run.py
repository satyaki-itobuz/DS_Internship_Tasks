# DEPENDENCIES
from src.supervised_hyperparameter import load_and_split_data
from config import hyperparameter_tuning_total_results
from src.supervised_hyperparameter import grid_search
from config import hyperparameter_tuning_top5_results
from config import stratified_sample_data
from config import STRATIFIED_DATA
from config import TARGET_COLUMN
from config import RANDOM_STATE
import pandas as pd
import itertools
import json

# Load and split the data
X_train, X_test, y_train, y_test = load_and_split_data(file_path     = STRTIFIED_DATA, 
                                                       target_column = TARGET_COLUMN, 
                                                       random_state  = RANDOM_STATE)
  
# Define the hyperparameter values
C_values = [0.01, 0.1, 1, 10, 100]
solvers = ['liblinear', 'sag', 'saga']
tol_values = [1e-4, 1e-3, 1e-2]

# Generate all possible combinations using itertools.product
param_combinations = itertools.product(C_values, solvers, tol_values)

# Store results for each combination
results = list()

for C, solver, tol in param_combinations:
    # Skip invalid combinations for solver
    if solver in ['liblinear']:# These solvers support L1
        # Create the parameter grid
        param_grid = {'penalty' : 'l1',  # Corrected 'penalty'
                      'C'       : C,
                      'solver'  : solver,
                      'tol'     : tol
                     }

        # Perform grid search for the current parameters
        result = grid_search(X_train    = X_train, 
                             X_test     = X_test, 
                             y_train    = y_train, 
                             y_test     = y_test, 
                             param_grid = param_grid)
        
        # Append the result in the results list
        results.append(result)

# Writing top-5 results in a json file
with open(file = HYPERPARAMETER_TUNING_TOTAL_RESULTS, mode = 'w') as file_pointer:
    json.dump(obj    = results, 
              fp     = file_pointer,
              indent = 4)
file_pointer.close()

# Print final summary sorted by test accuracy
print("\nFinal Summary (Top 5 Combinations):")
# Sort results by test accuracy in descending order
sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
# Display the top 5 combinations
for idx, res in enumerate(sorted_results[:5]):
    print(f"Rank {idx + 1}:")
    print(f"  Parameters: {res['params']}")
    print(f"  Test Accuracy: {res['test_accuracy']:.4f}")

top5_result_dataframe = pd.DataFrame(data = sorted_results[:5])
# Dump top5 results in a CSV table
top5_result_dataframe.to_csv(path_or_buf = HYPERPARAMETER_TUNING_TOP5_RESULTS., 
                             index       = False)

