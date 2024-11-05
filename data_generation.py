import pandas as pd
from sklearn.datasets import make_blobs, load_digits
from sklearn.decomposition import PCA


def generate_data():



    # generating using make blobs
    X, y = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=2.3,
        random_state=43
    )


    ''' 
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=2.3,
        random_state=26
    )
    '''


    # Tworzenie DataFrame z funkcjami i etykietami
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Label'] = y  # Dodanie prawdziwych etykiet do DataFrame
    return df
