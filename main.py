import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from joblib import dump
from custom_transformers import TextSelector, NumberSelector
import csv


# Function to convert pay range strings to average values
def convert_pay_range_to_avg(pay_str):
    if pd.isnull(pay_str) or pay_str.strip() == '':
        return np.nan
    if isinstance(pay_str, str):
        pay_str = ''.join(filter(str.isdigit or str.isspace or str == '.', pay_str))  # Keep only digits, spaces, and decimal points
        if pay_str.strip() == '' or not pay_str.replace('.', '', 1).isdigit():
            return np.nan
        if "-" in pay_str:
            low, high = pay_str.split("-")
            return (float(low) + float(high)) / 2
        else:
            return float(pay_str)
    else:
        return pay_str

data = []
with open('output1.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 9:  # Only process rows with exactly 9 elements
            data.append(row)

# Create DataFrame from data
df = pd.DataFrame(data, columns=['count_apps_id', 'job_id', 'job_title', 'job_description', 'min_pay', 'max_pay', 'commission', 'state', 'zip_code'])
# Convert the 'count_apps_id' column to numeric

# Convert the 'min_pay', 'max_pay', and 'commission' columns to numeric
for col in ['min_pay', 'max_pay', 'commission']:
    df[col] = df[col].replace('[\$,]', '', regex=True).apply(convert_pay_range_to_avg)
    df[col] = df[col].fillna(0)
# Define the numeric and text columns
numeric_columns = ['min_pay', 'max_pay', 'commission']
text_columns = ['job_title', 'job_description']

# Convert the 'count_apps_id' column to numeric
df['count_apps_id'] = pd.to_numeric(df['count_apps_id'], errors='coerce')
# Remove rows where 'count_apps_id' is NaN
df = df.dropna(subset=['count_apps_id'])

# Split the data into training and test sets
X = df[numeric_columns + text_columns]
y = df['count_apps_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines for text feature extraction
text_pipelines = []
for col in text_columns:
    text_pipelines.append(
        (col, Pipeline([
            ('selector', TextSelector(col)),  # Select the text column
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000))  # Convert text to TF-IDF features
        ]))
    )

# Create pipelines for numeric feature preprocessing
numeric_pipelines = []
for col in numeric_columns:
    numeric_pipelines.append(
        (col, Pipeline([
            ('selector', NumberSelector(col)),  # Select the numeric column
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by imputing with column mean
            ('scaler', StandardScaler())  # Scale the numeric features
        ]))
    )
# Create a feature union to combine
preprocessor = FeatureUnion(text_pipelines + numeric_pipelines)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MLPRegressor(shuffle=True, 
                                max_iter=1000, 
                                random_state=42,
                                early_stopping = True
                                ))  
])

# Define the hyperparameter search space
param_dist = {
    'classifier__hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
    'classifier__activation': ['relu', 'tanh', 'logistic'],
    'classifier__solver': ['adam', 'sgd'],
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
    'classifier__learning_rate': ['constant', 'adaptive', 'invscaling'],
    'classifier__batch_size': [32, 64, 128, 256],
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, n_iter=50, random_state=42)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

dump(best_model, 'job_posting_pipeline.pkl')
print("pipeline created with best model")

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")
print(f"Train R^2: {r2_train}")
print(f"Test R^2: {r2_test}")
