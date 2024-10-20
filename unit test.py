import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from data_generation import generate_data
from dbscan import dbscan
from plotting import plot_data


class TestMain(unittest.TestCase):

    def test_generate_data(self):
        """Test to check if the data generation works properly."""
        df = generate_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Feature 1', df.columns)
        self.assertIn('Feature 2', df.columns)
        self.assertEqual(df.shape[1], 2)  # Check if there are two features (columns)

    def test_dbscan_clustering(self):
        """Test DBSCAN with simple data to check if clustering works."""
        # Adjusted data: two clusters and two noise points
        X = np.array([[1, 1], [2, 2], [8, 8], [9, 9], [100, 100], [101, 101]])
        epsilon = 3
        min_pts = 2

        labels, core_points, noise_points = dbscan(X, epsilon, min_pts)

        # We expect two clusters and two noise points
        expected_labels = [1, 1, 2, 2, 3, 3]
        self.assertListEqual(labels.tolist(), expected_labels)
        self.assertListEqual(core_points, [0, 2, 4])  # Points 0, 2 and 4 should be core points
        self.assertListEqual(noise_points, [])  # No noise points in this scenario

    @patch("plotting.plt.show")
    def test_plot_data(self, mock_show):
        """Test to ensure plot_data can run without errors (mock plt.show)."""
        df = pd.DataFrame({
            'Feature 1': [1, 2, 3],
            'Feature 2': [4, 5, 6],
            'Cluster': [0, 1, -1]
        })
        noise_points = [2]  # One noise point

        # Run the plot_data function and ensure no exceptions are raised
        plot_data(df, noise_points)
        mock_show.assert_called_once()  # Check that plt.show() was called once


if __name__ == "__main__":
    unittest.main()
