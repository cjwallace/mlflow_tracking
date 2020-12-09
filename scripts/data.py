from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_features=20,      # generate 20 features
    n_informative=5,    # of which 5 are informative
    n_samples=1000,     # generate 1000 datapoints
    random_state=123    # generate the same data every time
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123)