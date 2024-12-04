# Required Dependencies
import numpy as np
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve


logger = logging.getLogger(__name__)

# Loading the preprocessed dataset
def load_dataset():
    """
    Loads the dataset
    Returns:
    - data (dataframe): the loaded the dataset
    """
    try:
        logger.info("Loading Dataset...")
        data = pd.read_csv("./data/preprocessed_data.csv")
        data = data.drop(columns=(["Unnamed: 0","order_id","product_id","user_id","days_since_prior_order_binned"]))
        logger.info(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading dataset {e}")


# Stratified Sampling on the Data
def stratified_sample(df, stratify_col, frac):
    """
    Performs stratified sampling on the dataset
    Args:
    - df (dataframe): dataset.
    - stratify_col (string): sampling based on the column.
    - frac (float): size of the sample.
    
    Returns:
    - stratified_df: stratified sampled data.
    """ 

    try:
        logger.info(f"Performing stratified sampling on column: {stratify_col} with fraction: {frac}")
        stratified_df = df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
        stratified_df = stratified_df.reset_index(drop=True)
        logger.info(f"Stratified sample data created with shape: {stratified_df.shape}")
        return stratified_df
    
    except Exception as e:
        logger.error(f"Error during stratified sampling: {e}")
        raise

# Splitting the data into training and testing set
def data_split(data):
    """
    Splits the data into training and testing set.
    Args:
    - data (dataframe): dataset
    
    Returns:
    - X_train (array-like): training set of feature matrix.
    - X_test (array-like): testing set of feature matrix.
    - y_train (array-like): training set of target vector.
    - y_test (array-like): testing set of target vector.
    """
    try:
        logger.info("Splitting the data into training and testing sets...")
        X = data.drop(columns="reordered")
        y = data["reordered"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        logger.info(f"Data split completed. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

# Initial Model Selection on sample data
def Initial_model_selection(models,X_train,y_train):
    """
    Performs initial model selection on a stratified 30% sample of the dataset using k-fold cross-validation.
    Args:
    - X_train (array-like): Feature matrix.
    - y_train (array-like): Target vector.
    - models (dict): Dictionary of models to evaluate.
    
    Returns:
    - results_df (dataframe): contains model_name, model and cross_val_score
    - best_models(dict): contains the best 4 models_name and the model
    """

    logger.info("Initial model selection")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    try:
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train.fillna(0), y_train, cv=kf, scoring='f1',n_jobs=-1)
            logger.info(f"{model} fitted with mean f1 score: {cv_scores.mean()}")
            results.append({
                'model_name': name,
                'model': model,
                'cross_val_score': cv_scores.mean()
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="cross_val_score",ascending=False,ignore_index=True)
        
        top_4_models = results_df.head(4)
        
        best_models = {}
        for _, row in top_4_models.iterrows():
            best_models[row['model_name']] = row['model']
        
        return results_df, best_models
    except Exception as e:
        logger.error(f"Error during initial model selection {e}")
        raise

# Final Model Selection on the initially selected best models 
def Final_model_selection(models, X_train, y_train):
    """
    Performs final model selection on whole data using k-fold cross-validation.
    Args:
    - X_train (array-like): Feature matrix.
    - y_train (array-like): Target vector.
    - models (dict): Dictionary of models to evaluate.
    
    Returns:
    - results_df (dataframe): contains model_name, model and cross_val_score
    - best_models(dict): contains the best model_name and the model
    """
    logger.info("Final model selection")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    try:
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train.fillna(0), y_train, cv=kf, scoring='f1',n_jobs=-1)
            logger.info(f"{model} fitted with mean f1 score: {cv_scores.mean()}")
            results.append({
                'model_name': name,
                'model': model,
                'cross_val_score': cv_scores.mean()
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="cross_val_score",ascending=False,ignore_index=True)
        
        top_model = results_df.head(1)
        
        best_model = {}
        for _, row in top_model.iterrows():
            best_model[row['model_name']] = row['model']
        
        return results_df, best_model
    
    except Exception as e:
        logger.error(f"Error during final model selection {e}")
        raise



# Final model training and evaluation
def model_evaluation(model, X_train,X_test, y_test, y_train):
    """
    Performs final model evaluation.
    Args:
    - model (onject): Final model.
    - X_train (array-like): Feature matrix training set.
    - X_test (array-like): Feature matrix testing set.
    - y_train (array-like): Target vector training test.
    - y_test (array-like): Target vector testing set.
    """
    logger.info(" Final model evaluation")
    lgb = model
    lgb.fit(X_train.fillna(0),y_train)
    y_pred = lgb.predict(X_test.fillna(0))

    logger.info(f"accuracy score: {accuracy_score(y_test,y_pred)}")
    logger.info(f"precision score: {precision_score(y_test,y_pred)}")
    logger.info(f"Recall score: {recall_score(y_test,y_pred)}")
    logger.info(f"f1 score: {f1_score(y_test,y_pred)}")

    final_result = {
        "model":model,
        "accuracy_score": accuracy_score(y_test,y_pred),
        "precision_score": precision_score(y_test,y_pred),
        "Recall_score": recall_score(y_test,y_pred),
        "f1 score": f1_score(y_test,y_pred)
    }

    logger.info("Saving the final model results.")
    try:
        with open('./results/initial_seletion_results.json', 'w') as fp: 
            # Make dictionary items JSON-serializable
            json.dump(final_result, fp, indent = 4)
        fp.close()
        logger.info("Saved the final model results.")
    except Exception as e:
        logging.error(f"Error saving final model results: {e}")
        raise


    # Confusion Matrix
    try:
        logger.info("saving confuion matrix.")
        fig=plt.figure()
        cm = confusion_matrix(y_test,y_pred,labels=lgb.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lgb.classes_)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("./plots/confusion_matrix.png")
        logger.info("Saved confusion matrix.")
    except Exception as e:
        logging.error(f"Error saving confusion matrix: {e}")
        raise

    # ROC-AUC Curve
    try:
        logger.info("saving ROC-AUC Curve.")
        lgb_roc_auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, lgb.predict_proba(X_test.fillna(0))[:,1])

        plt.figure()
        plt.plot(fpr, tpr, label = "AUC (area = %0.2f)" % lgb_roc_auc)
        plt.plot([0, 1], [0, 1],'g--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend()
        plt.savefig("./plots/ROC-AUC.png")
        logger.info("Saved ROC-AUC Curve.")
    except Exception as e:
        logging.error(f"Error saving ROC-AUC Curve: {e}")
        raise

    # Saving the final model
    try:
        logger.info("saving final model.")
        with open ('./results/best_model.pkl','wb') as file:
            pickle.dump(lgb,file)
        logger.info("Model saved in results directory")
    except Exception as e:
        logging.error(f"Error saving final model: {e}")
        raise



