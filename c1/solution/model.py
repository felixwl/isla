from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Model:
    def __init__(self, n_components=100):

        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.regression = LinearRegression()
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('regression', self.regression)
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)
