
import unittest
import os
import pandas as pd
from sklearn.exceptions import NotFittedError

class TestJobPostingPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('output.csv')  # Load the data
        self.model = None  # Placeholder for the model

    def test_pipeline_creation(self):
        # Run the script (you might need to modify this depending on how you've organized your code)
        exec(open("your_script.py").read())
        self.model = model  # The model variable should be available here if the script ran successfully
        self.assertIsNotNone(self.model, "Model was not created.")
        self.assertTrue(os.path.exists('job_posting_pipeline.pkl'), "Pipeline file was not created.")

    def test_model_predictions(self):
        if self.model is None:
            self.fail("Model is not available. Cannot test predictions.")
        else:
            X = self.df[['min_pay', 'max_pay', 'commission', 'job_title', 'job_description']]
            try:
                y_pred = self.model.predict(X)
                self.assertEqual(len(y_pred), len(self.df), "Number of predictions does not match number of samples.")
            except NotFittedError:
                self.fail("Model is not fitted. Cannot make predictions.")

if __name__ == '__main__':
    unittest.main()
  # Here's the change
