# main.py

from data_generation import generate_data
from dbscan import dbscan
from plotting import plot_data

# Generowanie danych
df = generate_data()

# Konwersja danych do tablicy NumPy
X = df[['Feature 1', 'Feature 2']].values

# Ustawienia parametrów DBSCAN
epsilon = 3
min_pts = 4

# Uruchomienie algorytmu DBSCAN
labels, core_points, noise_points = dbscan(X, epsilon, min_pts)

# Dodanie etykiet klastra do ramki danych
df['Cluster'] = labels

# Tworzenie wykresu
plot_data(df, noise_points)
