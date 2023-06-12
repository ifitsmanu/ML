# feature_extraction_preprocessing.py

import re
import string

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder, StandardScaler)


class lemmatize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda row: [self.lemmatizer.lemmatize(word) for word in row])



class TextVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, embeddings, pca_components):
        self.embeddings = embeddings
        self.pca_components = pca_components
        self.pca = PCA(n_components=self.pca_components)


    def fit(self, X, y=None):
        # Fit the PCA model to the embeddings
       
        all_embeddings = [self.embeddings[word] for word in self.embeddings.key_to_index.keys()]
        self.pca.fit(all_embeddings)
        return self

    def transform(self, X, y=None):
        result = []
        for text in X:
            vector = np.zeros(self.pca_components)  # Initialize vector with dimensions equal to PCA components
            count = 0
            
            for word in str(text).split():

                if word in self.embeddings:
                    word_embedding = self.embeddings[word]
                    reduced_embedding = self.pca.transform(word_embedding.reshape(1, -1))
                    vector += reduced_embedding.squeeze()
                    count += 1
            if count > 0:
                vector /= count
            result.append(vector)
        return np.array(result)



def clean_html_tags(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    return cleantext

def convert_to_float(value):
    if isinstance(value, str):
        # Extract numbers from the string using regex
        numbers = re.findall(r'\d+(?:\.\d+)?', value)
        if numbers:
            # Return the average of the extracted numbers
            return sum(float(num) for num in numbers) / len(numbers)
        else:
            return 0.0
    return value





def preprocess_data(df, embeddings, embedding_size):
    """
    Preprocesses a DataFrame for machine learning.

    Args:
        df: The DataFrame to preprocess.
        embeddings: A set of embeddings.
        embedding_size: The size of the embeddings.

    Returns:
        The preprocessed DataFrame and fitted preprocessor.
    """
    if 'commission' in df.columns:
        df['commission'] = df['commission'].astype(str).str.replace('%', '').astype(float)

    # Separate numerical and text data
    num_df = df[['min_pay', 'max_pay', 'commission']]
    text_df = df[['job_title', 'job_description']]
    
    # Preprocessing steps
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    txt_transformer = Pipeline([
       ('vectorizer', TextVectorizer(embeddings, 50))
    ])

    # Preprocess numerical data
    num_df_transformed = num_transformer.fit_transform(num_df)

    # Preprocess text data
    text_df_transformed = txt_transformer.fit_transform(text_df)

    # Concatenate preprocessed numerical and text data
    X_transformed = np.concatenate([num_df_transformed, text_df_transformed], axis=1)

    return X_transformed
