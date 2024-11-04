import pandas as pd
from sklearn.datasets import make_blobs


def generate_data():
    # Generate synthetic data with labels
    X, y = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=2.3,
        random_state=43
    )
    # Create a DataFrame with features and labels
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Label'] = y  # Add the true labels to the DataFrame
    return df

