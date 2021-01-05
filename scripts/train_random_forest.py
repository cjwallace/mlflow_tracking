import argparse

import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scripts.data import X_train, X_test, y_train, y_test

parser = argparse.ArgumentParser()
parser.add_argument('--max-depth', type=int, default=3)
parser.add_argument('--n-estimators', type=int, default=20)
args = parser.parse_args()


with mlflow.start_run():

    # Log hyperparameters for the training run

    mlflow.log_params({
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators
    })


    # Define and train a ML pipeline

    scaler = StandardScaler()

    rf = RandomForestClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators
    )

    pipe = make_pipeline(scaler, rf)
    pipe.fit(X_train, y_train)


    # Log the model performance metrics, and save the serialized model

    mlflow.log_metrics({
        'train_accuracy': pipe.score(X_train, y_train),
        'test_accuracy': pipe.score(X_test, y_test)
    })

    mlflow.sklearn.log_model(pipe,'models')