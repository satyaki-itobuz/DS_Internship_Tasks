# DEPENDENCIES
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings

# Ignore all the warnings during runtime
warnings.filterwarnings(action = 'ignore')


# REQUIRED FUNCTIONAITIES
def load_and_split_data(file_path, target_column, test_size=0.2, random_state=42):
    """
    Loads a dataset from a CSV file, preprocesses it, and splits it into train and test sets.
    """
    # Step 1: Load the dataset
    data = pd.read_csv(file_path)

    # Step 2: Handle categorical variables
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        le        = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Step 3: Define features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Step 4: Perform train-test split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def grid_search(X_train, X_test, y_train, y_test, param_grid):
    """
    Performs grid search for a single set of parameters and evaluates the model.
    """
    model = LogisticRegression(**param_grid)
    model.fit(X_train, y_train)

    # Calculate accuracy on test data
    y_pred        = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Parameters: {param_grid}")
    print(f"Test Accuracy: {test_accuracy}")

    grid_search_result = {'params'        : param_grid,
                          'test_accuracy' : test_accuracy,
                         }
    
    return grid_search_result

"""
if __name__ == "__main__":
    # Path to your CSV file and target column
    file_path = "/content/drive/MyDrive/filename.csv"  # Replace with your dataset path
    target_column = "reordered"  # Replace with your target column name

    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(file_path, target_column)

    # Define the hyperparameter values
    C_values = [0.01, 0.1, 1, 10, 100]
    solvers = ['liblinear', 'sag', 'saga']
    tol_values = [1e-4, 1e-3, 1e-2]
   # max_iter_values = [100, 200, 500]
    #class_weights = [None, 'balanced']

    # Generate all possible combinations using itertools.product
    param_combinations = itertools.product(C_values, solvers, tol_values)

    # Store results for each combination
    results = []

    for C, solver, tol in param_combinations:
        # Skip invalid combinations for solver
        if solver in ['liblinear']:  # These solvers support L1
            param_grid = {
                'penalty': 'l1',  # Corrected 'penalty'
                'C': C,
                'solver': solver,
                'tol': tol
            }

        # Perform grid search for the current parameters
        result = grid_search(X_train, X_test, y_train, y_test, param_grid)
        results.append(result)

    # Print final summary sorted by test accuracy
print("\nFinal Summary (Top 5 Combinations):")

# Sort results by test accuracy in descending order
sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

# Display the top 5 combinations
for idx, res in enumerate(sorted_results[:5]):
    print(f"Rank {idx + 1}:")
    print(f"  Parameters: {res['params']}")
    print(f"  Test Accuracy: {res['test_accuracy']:.4f}")

"""