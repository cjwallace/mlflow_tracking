import argparse

import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from scripts.data import X_train, X_test, y_train, y_test

parser = argparse.ArgumentParser()
parser.add_argument('--n-neighbors', type=float, default=5)
args = parser.parse_args()


with mlflow.start_run():

    # Log hyperparameters for the training run

    mlflow.log_param('n_neighbors', args.n_neighbors)


    # Define and train a ML pipeline

    scaler = StandardScaler()
    kn = KNeighborsClassifier(args.n_neighbors)

    pipe = make_pipeline(scaler, kn)
    pipe.fit(X_train, y_train)


    # Log the model performance metrics, and save the serialized model

    mlflow.log_metrics({
        'train_accuracy': pipe.score(X_train, y_train),
        'test_accuracy': pipe.score(X_test, y_test)
    })

    mlflow.sklearn.log_model(pipe, 'models')