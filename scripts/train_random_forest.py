import mlflow

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scripts.data import X_train, X_test, y_train, y_test

MAX_DEPTH = 3
N_ESTIMATORS = 20

with mlflow.start_run():

    mlflow.log_params({
      'max_depth': MAX_DEPTH,
      'n_estimators': N_ESTIMATORS
    })
  
    # A random forest classifier.
    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)

    pipe = make_pipeline(scaler, rf)
    pipe.fit(X_train, y_train)

    mlflow.log_metrics({
      'train_accuracy': pipe.score(X_train, y_train),
      'test_accuracy': pipe.score(X_test, y_test)
    })

    mlflow.sklearn.log_model(pipe,'models')