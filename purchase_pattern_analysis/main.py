from src.model_selection_and_evaluation import load_dataset, stratified_sample, data_split
from src.model_selection_and_evaluation import Initial_model_selection, Final_model_selection, model_evaluation
import xgboost
import lightgbm
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def main():
    # Loading the dataset
    data = load_dataset()

    # Sampling the data for initial model selection
    sampled_data = stratified_sample(data,'reordered',0.3)
    # Train_Test_Split on the sampled data
    X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = data_split(sampled_data)

    # Initializing the base models
    models = {
        "Logistic Regression with l2 regularization": LogisticRegression(solver='saga',max_iter=1500,penalty='l2',n_jobs=-1),
        "Logistic Regression with l1 regularization": LogisticRegression(solver='saga',max_iter=1500,penalty='l1',n_jobs=-1),
        "Logistic Regression with elastic net regularization": LogisticRegression(solver='saga',max_iter=1500,penalty='elasticnet',n_jobs=-1,l1_ratio=0.5),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(max_depth=1),
        "Naive Bayes": BernoulliNB(),
        "Random Forest": RandomForestClassifier(max_depth=3,n_jobs=-1,n_estimators=50),
        "xgboost": xgboost.XGBClassifier(n_jobs=-1,max_depth=3,n_estimators=50),
        "lightGBM": lightgbm.LGBMClassifier(verbose=-1,n_jobs=-1)
    }

    # Initial model selection on the sample data
    initial_results,best_models = Initial_model_selection(models,X_train=X_train_sampled,y_train=y_train_sampled)

    with open('./results/initial_seletion_results.json', 'w') as fp: 
        # Make dictionary items JSON-serializable
        json_data = {key : value.to_json() for key, value in initial_results.items()}
        json.dump(json_data, fp, indent = 4)
    fp.close()

    # Train_Test_Split on the sampled data
    X_train, X_test, y_train, y_test = data_split(data)

    # Final model selection on the whole data
    final_results, best_model = Final_model_selection(models=best_models,X_train=X_train, y_train=y_train)

    with open('./results/final_selection_results.json', 'w') as fp: 
        # Make dictionary items JSON-serializable
        json_data = {key : value.to_json() for key, value in final_results.items()}
        json.dump(json_data, fp, indent = 4)
    fp.close()
    
    model = list(best_model.values())[0]

    # final model evaluation
    model_evaluation(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    main()