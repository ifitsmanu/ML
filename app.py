import os

import pandas as pd
from flask import Flask, jsonify, request
from joblib import load
from custom_transformers import TextSelector, NumberSelector
from job_post_generator import generate_job_post

current_dir = os.getcwd()
app = Flask(__name__)
app.debug = True
job_posting_pipeline_path = os.path.join(current_dir, 'job_posting_pipeline.pkl')
pipeline = load(job_posting_pipeline_path)


def performance_score(predicted_applicants):
    if predicted_applicants >= 100:
        return 'High'
    elif predicted_applicants >= 50:
        return 'Medium'
    else: 
        return 'Low'
    
@app.route('/predict', methods=['POST'])
def predict_applicants():
    pipeline = load(job_posting_pipeline_path)
    data = request.get_json()
    
    job_title = data['job_title']
    job_description = data['job_description']
    min_pay = data['min_pay']
    max_pay = data['max_pay']
    commission = data['commission']
    
    input_data = pd.DataFrame({
        'job_title': [job_title], 
        'job_description': [job_description],
        'min_pay': [min_pay],
        'max_pay': [max_pay],
        'commission': [commission]
    })
    predicted_applicants = pipeline.predict(input_data)[0]
    
    description_length = len(job_description)
    score = performance_score(predicted_applicants)
    
    response= {
        'predicted_applicants': predicted_applicants,
        'performance_score': score
    }
    
    return jsonify(response)

@app.route('/generate', methods=['POST'])
def generate_post():
    data = request.get_json()

    job_title = data['job_title']

    # Load the dataset (you might want to do this once, not for every request)
    df = pd.read_csv('job_postings.csv')

    # Check if the job title is in the dataset
    for title in df['job_title']:
        if job_title.lower() in title.lower():
            generated_post = generate_job_post(title)
            break
    else:
        return jsonify({'error': 'Job title not found in the dataset'}), 404

    response = {
        'generated_job_post': generated_post
    }

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
    
    
