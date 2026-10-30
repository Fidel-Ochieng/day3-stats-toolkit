import unittest
import tempfile
import csv 
from pathlib import Path 


from stats_toolkit import (
    mean, median, mode, variance, std_dev,
    quartiles, iqr_outliers, load_column
    )

"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([1, 2, 3]), 2)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])

    def test_empty_input_raises(self):
       with self.assertRaises(ValueError):
            mean([])


class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        self.assertEqual(variance([1, 2, 3], sample=False),2 / 3)
        self.assertEqual(variance([1, 2, 3], sample=True), 1)


    def test_std_dev_single_value(self):
        with self.assertRaises(ValueError):
            std_dev([5], sample=True)


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([5, 5, 5, 5, 5])
        self.assertEqual(result["outliers"], [])

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])
        self.assertIn(100, result["outliers"])


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["daily_deposits_kes"])
                writer.writerow(["100"])
                writer.writerow([""])
                writer.writerow(["n/a"])
                writer.writerow(["200"])


            values, rejected = load_column(
                csv_path, "daily_deposits_kes"
            )

            self.assertEqual(values, [100.0, 200.0])
            self.assertEqual(len(rejected), 2)
            

if __name__ == "__main__":
    unittest.main()