# data_generation.py

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def generate_data():
    X, _ = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=2.3,
        random_state=42
    )
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    return df

'''

def generate_data():
    
    digits = load_digits()
    X = digits.data  

    # pca reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame(X_pca, columns=['Feature 1', 'Feature 2'])

    return df

'''