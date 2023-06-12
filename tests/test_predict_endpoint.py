import unittest
import requests
import json

class TestPredictEndpoint(unittest.TestCase):
    def test_predict(self):
        url = "http://127.0.0.1:5000/predict"  # replace with your server's URL
        data = {
            "job_title": "Software Engineer",
            "job_description": "We are looking for a skilled software engineer with experience in Python and web development...",
            "min_pay": 50000,  # replace with appropriate value
            "max_pay": 100000,  # replace with appropriate value
            "commission": 5000  # replace with appropriate value
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=json.dumps(data), headers=headers)
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertIn('predicted_applicants', response_data)
        self.assertIn('performance_score', response_data)

if __name__ == '__main__':
    unittest.main()
