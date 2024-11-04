import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(df, noise_points=None, cluster_column_custom='Cluster_custom', cluster_column_kmeans='Cluster_kmeans'):
    # Ustawienia wykresu
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Wykres lewy - dane wygenerowane z etykietami
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue='Label', data=df, 
        palette='viridis', s=60, ax=axes[0], legend=False
    )
    axes[0].set_title("True Labels (Generated Data)")
    
    # Wykres środkowy - wyniki klasteryzacji własnej implementacji DBSCAN
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

    # Wykres prawy - wyniki klasteryzacji K-Means
    sns.scatterplot(
        x='Feature 1', y='Feature 2', hue=cluster_column_kmeans, data=df, 
        palette='tab10', s=60, ax=axes[2], legend=False
    )
    axes[2].set_title("Clusters (K-Means Result)")
    axes[2].set_xlabel("Feature 1")
    axes[2].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()
