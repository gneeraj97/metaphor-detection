import pandas as pd
import nltk
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix, hstack
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import argparse
import joblib


def find_first_occurrence(keyword, sentences):
    for i, sentence in enumerate(sentences):
        if keyword.lower() in sentence.lower():
            return i
        elif keyword.lower() +"s" in sentence.lower():
            return i
    return -1


def train(path_to_csv):
    # Input: Path to dataset
    # Output: trained_model
    nltk.download('punkt')
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path_to_csv)

    # Tokenize the 'text' column and create a new column 'sentences'
    df['sentences'] = df['text'].apply(sent_tokenize)

    # Mapping of metaphor IDs to keywords
    keyword_mapping = {
        0: "road",
        1: "candle",
        2: "light",
        3: "spice",
        4: "ride",
        5: "train",
        6: "boat"
    }

    # Apply the mapping to create a new 'keyword' column
    df['keyword'] = df['metaphorID'].map(keyword_mapping)

    # Drop the 'metaphorID' column
    df.drop("metaphorID", inplace=True, axis=1)

    # Apply a function to create a new column for the first occurrence index of the keyword in sentences
    df['first_occurrence_index'] = df.apply(lambda row: find_first_occurrence(row['keyword'], row['sentences']), axis=1)

    # Create columns for the sentence with the first occurrence, sentence before, and sentence after
    df['first_occurrence_sentence'] = df.apply(lambda row: row['sentences'][row['first_occurrence_index']], axis=1)
    df['sentence_before'] = df.apply(lambda row: row['sentences'][row['first_occurrence_index'] - 1] if row['first_occurrence_index'] > 0 else '', axis=1)
    df['sentence_after'] = df.apply(lambda row: row['sentences'][row['first_occurrence_index'] + 1] if row['first_occurrence_index'] < len(row['sentences']) - 1 else '', axis=1)

    # Create a CountVectorizer with specified parameters
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    # Fit and transform the text data using CountVectorizer
    dtm = cv.fit_transform(df['text'])

    # Create a Latent Dirichlet Allocation model with 10 components
    LDA = LatentDirichletAllocation(n_components=10, random_state=42)
    LDA.fit(dtm)

    # Transform the entire document-term matrix to get topic results for the whole text
    topic_results_whole = LDA.transform(dtm)
    df['full_topic'] = topic_results_whole.argmax(axis=1)

    # Transform the target sentences and their surrounding sentences to get topic results
    target = cv.transform(df['first_occurrence_sentence'])
    target_prev = cv.transform(df['sentence_before'])
    target_next = cv.transform(df['sentence_after'])

    # Transform the target sentences and their surrounding sentences to get topic results
    topic_results_target = LDA.transform(target)
    topic_results_target_prev = LDA.transform(target_prev)
    topic_results_target_next = LDA.transform(target_next)

    # Create new columns for target topics
    df['target_topic'] = topic_results_target.argmax(axis=1)
    df['target_prev_topic'] = topic_results_target_prev.argmax(axis=1)
    df['target_next_topic'] = topic_results_target_next.argmax(axis=1)

    # Select relevant columns for input data
    input_data = df[['first_occurrence_sentence', 'sentence_before', 'sentence_after', 'full_topic', 'target_topic', 'target_prev_topic', 'target_next_topic']]

    text_features = cv.transform(input_data['first_occurrence_sentence'])
    text_features_before = cv.transform(input_data['sentence_before'])
    text_features_after = cv.transform(input_data['sentence_after'])

    # Concatenate text features with other categorical features
    X = hstack([text_features, text_features_before, text_features_after, input_data[['full_topic', 'target_topic', 'target_prev_topic', 'target_next_topic']]])

    # Target variable
    # Map 'False' to 0 and 'True' to 1 in the 'label_boolean' column
    y_train = df['label_boolean'].map({False: 0, True: 1})
    X_train = pd.DataFrame(X.todense(), columns=range(X.shape[1]))
    #print(X_train.shape)



    ##################### Model ########################

    # Best hyperparameters
    params = {
        'colsample_bytree': 0.7977426414368413,
        'learning_rate': 0.10445809047577313,
        'max_depth': 5,
        'n_estimators': 100,
        'subsample': 0.7130026818171501
    }

    # Initialize XGBoost classifier with the specified hyperparameters
    xgb_classifier = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42  # Set a random state for reproducibility
    )

    # Train the XGBClassifier
    xgb_classifier.fit(X_train, y_train)

    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(xgb_classifier, model_file)

    with open('trained_LDA.pkl', 'wb') as model_file:
        pickle.dump(LDA, model_file)

    # Save the vectorizer
    joblib.dump(cv, 'vectorizer.joblib')
    joblib.dump(cv.vocabulary_, 'vocabulary.joblib')

    return xgb_classifier

# Command line argument parsing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and save XGBoost model")
    parser.add_argument("path_to_csv", help="Path to the input CSV file")

    args = parser.parse_args()

    # Train and save the model
    trained_model = train(args.path_to_csv)
    print("Model trained and saved successfully in the current working directory along with LDA, vocabulary and vectorizer.")


