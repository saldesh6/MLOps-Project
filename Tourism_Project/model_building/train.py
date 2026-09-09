# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")   
mlflow.set_experiment("Tourism_Package_Prediction")     

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar", 
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", 
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]   

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation", "ProductPitched"
]   

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators':,        
    'xgbclassifier__max_depth':,           
    'xgbclassifier__colsample_bytree': [0.8, 1.0],    
    'xgbclassifier__colsample_bylevel': [0.8, 1.0],   
    'xgbclassifier__learning_rate': [0.01, 0.1],       
    'xgbclassifier__reg_lambda': [1.0, 10.0],          
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.5    

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Case-insensitive deployment path fallback mapping setup
    if os.path.exists("Tourism_Project/deployment"):
        model_path = "Tourism_Project/deployment/best_model.joblib"
    elif os.path.exists("tourism_project/deployment"):
        model_path = "tourism_project/deployment/best_model.joblib"
    else:
        os.makedirs("Tourism_Project/deployment", exist_ok=True)
        model_path = "Tourism_Project/deployment/best_model.joblib"

    joblib.dump(best_model, model_path)  
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved successfully to {model_path}")
