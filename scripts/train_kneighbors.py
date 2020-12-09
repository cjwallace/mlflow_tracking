import mlflow

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from scripts.data import X_train, X_test, y_train, y_test

N_NEIGHBORS = 5

with mlflow.start_run():
    mlflow.log_param('n_neighbors', N_NEIGHBORS)

    # A nearest neighbor classifier.
    scaler = StandardScaler()
    kn = KNeighborsClassifier(N_NEIGHBORS)

    pipe = make_pipeline(scaler, kn)
    pipe.fit(X_train, y_train)

    mlflow.log_metrics({
      'train_accuracy': pipe.score(X_train, y_train),
      'test_accuracy': pipe.score(X_test, y_test)
    })

    mlflow.sklearn.log_model(pipe,'models')