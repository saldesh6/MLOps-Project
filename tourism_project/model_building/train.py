import pandas as pd
import xgboost as xgb
import joblib
import mlflow
import os
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]
categorical_features = ["TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation", "ProductPitched"]

# FIXED: Explicit numeric indexers force a single decimal scalar value
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators':[50, 100, 150, 200],
    'xgbclassifier__max_depth':[2, 3, 5, 7],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    output_dir = "tourism_project/deployment"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.joblib")
    
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✓ Training routine finalized. Binary package saved to: {model_path}")
