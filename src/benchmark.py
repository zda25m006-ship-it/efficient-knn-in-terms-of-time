import time
from sklearn.neighbors import KNeighborsClassifier


def benchmark_custom(model, X_train, y_train, X_test):
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, time.time() - start


def benchmark_sklearn(X_train, y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=5)
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, time.time() - start
