
import datetime  # Added line

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import TensorBoard  # Added line
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import \
    KerasClassifier as KerasClassifierOriginal


class KerasClassifier(KerasClassifierOriginal):
    def fit(self, x, y, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        return super().fit(x, y, **kwargs)

def create_neural_network_model(input_dim):
    # Create a simple feed-forward neural network model
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def create_model_wrapper(input_dim):
    def create_model():
        return create_neural_network_model(input_dim=input_dim)
    return create_model


def train_and_evaluate_nn_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def create_pipeline(preprocessor, model, input_dim):
    # Create a pipeline that first transforms the data using the preprocessor
    # and then trains the model
    model = create_model_wrapper(input_dim)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KerasClassifier(build_fn=model, epochs=10, batch_size=32, verbose=1))
    ]) 

    return pipeline






def run_pipeline(preprocessor, model_creator, X_train_transformed, y_train, X_test_transformed, y_test):
    # Determine the number of features after preprocessing
    max_features = X_train_transformed.shape[1]
    print("this hit 0")

    # Use this number to create the model with the correct input_dim
    model = model_creator(max_features)
    print("this hit 1")

    # Create a new pipeline with the already fit preprocessor and the correctly dimensioned model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KerasClassifier(build_fn=model, epochs=10, batch_size=32, verbose=1))
    ])
    print("this hit 2")

    # Fit this pipeline, which will only fit the model
    pipeline.fit(X_train_transformed, y_train)
    print("this hit 3")

    # Predict the test data using the transformed test data
    y_pred = pipeline.predict(X_test_transformed)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
