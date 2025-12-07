import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import mlflow
from huggingface_hub import HfApi

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi() # Instantiate HfApi

# Load HF-processed splits
Xtrain = pd.read_csv("hf://datasets/greatlearninglokeshrajoria/tourism-package-prediction/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/greatlearninglokeshrajoria/tourism-package-prediction/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/greatlearninglokeshrajoria/tourism-package-prediction/ytrain.csv").squeeze()
ytest = pd.read_csv("hf://datasets/greatlearninglokeshrajoria/tourism-package-prediction/ytest.csv").squeeze()

cat_cols = ['TypeofContact', 'Occupation', 'ProductPitched',
            'MaritalStatus', 'Designation', 'Gender']

# Preprocessing
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    remainder='passthrough'
)

# ⭐ FIX 1: Use CLASSIFIER instead of REGRESSOR
clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# Pipeline
pipeline = make_pipeline(preprocessor, clf)

# Hyperparameters (classifier version)
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10],
}

with mlflow.start_run():

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_

    # Predictions
    ypred = best_model.predict(Xtest)
    yprob = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(ytest, ypred),
        "precision": precision_score(ytest, ypred),
        "recall": recall_score(ytest, ypred),
        "f1": f1_score(ytest, ypred),
        "auc": roc_auc_score(ytest, yprob)
    }

    mlflow.log_metrics(metrics)
    mlflow.log_params(grid.best_params_)

    # Save model
    model_path = "tourism_pkg_predition_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    # Upload to HF
    repo_id = "greatlearninglokeshrajoria/tourism-package-prediction"
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model"
    )

    print("Model uploaded successfully to HuggingFace.")
