import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import pickle
from nltk.tokenize import sent_tokenize
import argparse
import nltk
import joblib
import warnings
warnings.filterwarnings("ignore")

def find_first_occurrence(keyword, sentences):
    for i, sentence in enumerate(sentences):
        if keyword.lower() in sentence.lower():
            return i
        elif keyword.lower() + "s" in sentence.lower():
            return i
    return -1

def predict(word, text):

    punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Load the trained model
    with open('trained_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('trained_LDA.pkl', 'rb') as model_file:
        LDA = pickle.load(model_file)

    # Load the vectorizer
    vectorizer = joblib.load('vectorizer.joblib')

    # Load the vocabulary
    vocabulary = joblib.load('vocabulary.joblib')

    # Set the vocabulary in the vectorizer
    vectorizer.vocabulary_ = vocabulary
    # Read the CSV file into a DataFrame
    test_data = pd.DataFrame({"keyword":[word], "text":[text]})

    # Tokenize the 'text' column and create a new column 'sentences'
    test_data['sentences'] = test_data['text'].apply(lambda x: punkt_tokenizer.tokenize(x))

    # Mapping of metaphor IDs to keywords
    # keyword_mapping = {
    #     0: "road",
    #     1: "candle",
    #     2: "light",
    #     3: "spice",
    #     4: "ride",
    #     5: "train",
    #     6: "boat"
    # }

    # # Apply the mapping to create a new 'keyword' column
    # test_data['keyword'] = test_data['metaphorID'].map(keyword_mapping)

    # Drop the 'metaphorID' column
    # test_data.drop("metaphorID", inplace=True, axis=1)

    # Apply a function to create a new column for the first occurrence index of the keyword in sentences
    test_data['first_occurrence_index'] = test_data.apply(lambda row: find_first_occurrence(row['keyword'], row['sentences']), axis=1)

    # Create columns for the sentence with the first occurrence, sentence before, and sentence after
    test_data['first_occurrence_sentence'] = test_data.apply(
        lambda row: row['sentences'][row['first_occurrence_index']], axis=1)
    test_data['sentence_before'] = test_data.apply(
        lambda row: row['sentences'][row['first_occurrence_index'] - 1] if row['first_occurrence_index'] > 0 else '', axis=1)
    test_data['sentence_after'] = test_data.apply(
        lambda row: row['sentences'][row['first_occurrence_index'] + 1] if row['first_occurrence_index'] < len(
            row['sentences']) - 1 else '', axis=1)

    # Set the vocabulary in the vectorizer
    vectorizer.vocabulary_ = vocabulary

    # Assuming test_data is your test data
    text_features = vectorizer.transform(test_data['first_occurrence_sentence'])
    text_features_before = vectorizer.transform(test_data['sentence_before'])
    text_features_after = vectorizer.transform(test_data['sentence_after'])

    # Transform the target sentences and their surrounding sentences to get topic results
    topic_results_target = LDA.transform(text_features)
    topic_results_target_prev = LDA.transform(text_features_before)
    topic_results_target_next = LDA.transform(text_features_after)

    # Transform the entire document-term matrix to get topic results for the whole text
    dtm = vectorizer.transform(test_data['text'])
    topic_results_whole = LDA.transform(dtm)
    test_data['full_topic'] = topic_results_whole.argmax(axis=1)

    # Create new columns for target topics
    test_data['target_topic'] = topic_results_target.argmax(axis=1)
    test_data['target_prev_topic'] = topic_results_target_prev.argmax(axis=1)
    test_data['target_next_topic'] = topic_results_target_next.argmax(axis=1)

    # Select relevant columns for input data
    # input_data = test_data[['first_occurrence_sentence', 'sentence_before', 'sentence_after', 'full_topic', 'target_topic', 'target_prev_topic', 'target_next_topic']]


    # Concatenate text features with other categorical features
    X_test = hstack([text_features, text_features_before, text_features_after,
                     test_data[['full_topic', 'target_topic', 'target_prev_topic', 'target_next_topic']]])
    
    X_test = pd.DataFrame(X_test.todense(), columns=range(X_test.shape[1]))

    # Predict using the loaded model
    y_pred = loaded_model.predict(X_test)

    # Map 0 to False and 1 to True
    y_pred_boolean = y_pred.astype(bool)

    # Save the DataFrame with the new column
    # pred_df = pd.DataFrame(y_pred_boolean, columns = ["label_boolean"])
    # output_path = 'predicted_labels.csv'
    # print("Few of the predicted values are printed below.")
    # print(pred_df.head(20))
    # print("NOTE: All predictions are saved in the predicted_labels.csv file")
    # pred_df.to_csv(output_path, index=False)

    return "Its a Metaphor!" if y_pred_boolean == 1 else "Its not a Metaphor!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and save test data with boolean labels.")
    parser.add_argument("word", type=str, help="Metaphor word")
    parser.add_argument("text", type=str, help="Text")
    args = parser.parse_args()

    result = predict(args.word, args.text)
    print(result)


