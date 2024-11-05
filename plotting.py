import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_data(df, noise_points=None, cluster_column_custom='Cluster_custom', cluster_column_kmeans='Cluster_kmeans', cluster_column_dbscan_sklearn='Cluster_dbscan_sklearn', cluster_column_hdbscan='Cluster_hdbscan'):
    # Ustawienia wykresu
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    
    # Wykres pierwszy - dane wygenerowane z etykietami
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue='Label', data=df, 
        palette='viridis', s=60, ax=axes[0], legend=False
    )
    axes[0].set_title("True Labels (Generated Data)")
    
    # Wykres drugi - wyniki klasteryzacji własnej implementacji DBSCAN
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue=cluster_column_custom, data=df, 
        palette='tab10', s=60, ax=axes[1], legend=False
    )
    if noise_points is not None:
        sns.scatterplot(
            x=df.loc[noise_points, 'Feature 1'],
            y=df.loc[noise_points, 'Feature 2'],
            color='red', marker='x', s=50, ax=axes[1], label="Noise"
        )
    axes[1].set_title("Clusters (Custom DBSCAN Result)")

    # Wykres trzeci - wyniki klasteryzacji K-Means
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue=cluster_column_kmeans, data=df,
        palette='tab10', s=60, ax=axes[2], legend=False
    )
    axes[2].set_title("Clusters (K-Means Result)")
    axes[2].set_xlabel("Feature 1")
    axes[2].set_ylabel("Feature 2")

    # Wykres czwarty - wyniki DBSCAN z scikit-learn
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue=cluster_column_dbscan_sklearn, data=df,
        palette='tab10', s=60, ax=axes[3], legend=False
    )
    axes[3].set_title("Clusters (scikit-learn DBSCAN Result)")
    axes[3].set_xlabel("Feature 1")
    axes[3].set_ylabel("Feature 2")

    # Wykres piąty - wyniki klasteryzacji HDBSCAN
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue=cluster_column_hdbscan, data=df,
        palette='tab10', s=60, ax=axes[4], legend=False
    )
    axes[4].set_title("Clusters (HDBSCAN Result)")

    plt.tight_layout()
    plt.show()
