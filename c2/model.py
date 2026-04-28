import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


def temporal_variance_log(X):
    # X shape: (n_epochs, 64, T) -> output: (n_epochs, 64)
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


class Model:
    def __init__(self):
        self.pipeline = Pipeline([
            ('log_var', FunctionTransformer(temporal_variance_log)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=1000)),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)
