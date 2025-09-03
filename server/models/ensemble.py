import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)

# HybridEnsemble for mixed scaling (LogisticRegression uses scaled features)
from sklearn.linear_model import LogisticRegression
class HybridEnsemble(EnsembleModel):
    def __init__(self, models, scaler=None):
        super().__init__(models)
        self.scaler = scaler
    def predict_proba(self, X):
        probas = []
        for model in self.models:
            if isinstance(model, LogisticRegression) and self.scaler is not None:
                probas.append(model.predict_proba(self.scaler.transform(X)))
            else:
                probas.append(model.predict_proba(X))
        avg_proba = np.mean(probas, axis=0)
        return avg_proba