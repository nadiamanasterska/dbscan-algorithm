# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df, noise_points):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Feature 1', y='Feature 2', hue='Cluster', data=df, palette='viridis')

    noise_points_df = df.iloc[noise_points]
    if len(noise_points) > 0:
        plt.scatter(noise_points_df['Feature 1'], noise_points_df['Feature 2'], marker='x', color='red', s=100, label='Noise Points')

    plt.title('Feature 1 vs. Feature 2 with DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
    plt.show()
