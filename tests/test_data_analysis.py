import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
print("sys.path:", sys.path)  # Debugging line

from src.data_analysis import analyze_data

class TestDataAnalysis(unittest.TestCase):
    def test_analyze_data(self):
        # Test analyzing a valid dataset
        cleaned_data = {'some': 'cleaned data'}
        analysis_results = analyze_data(cleaned_data)
        self.assertIsNotNone(analysis_results)
        # ...additional tests...

if __name__ == '__main__':
    unittest.main()
