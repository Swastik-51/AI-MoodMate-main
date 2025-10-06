import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer that cleans text by:
    - Converting to lowercase
    - Removing non-alphanumeric characters except spaces
    - Normalizing whitespace
    """
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        cleaned = []
        for t in X:
            t = str(t).lower()
            t = re.sub(r"[^a-z0-9\s]+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned
