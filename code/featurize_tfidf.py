import yaml
import pandas as pd
import os
import json
import joblib
import logging
import inspect
import re
import gc
import numpy
import sys
import numpy

from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


def load_clean_data(path):
    data = pd.read_csv(path)
    return data


def get_tfidf(df):
    #### embed all documents with tfidf
    pprint('=' * 20 + 'Embedding with tfidf' + '=' * 20)
    ngram_range = (1, 3)
    vectorizer = TfidfVectorizer(
        max_features = 300,
        sublinear_tf=True,
        strip_accents='unicode',
        stop_words='english',
        analyzer='word',
        token_pattern=r'\w{3,}',
        norm='l2',
        max_df=.9
        )
    df['text'] = list(vectorizer.fit_transform(df['text'].values.astype('U')).toarray())
    return df


def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)
    pprint(df)


logging.basicConfig(level=logging.DEBUG)
numpy.set_printoptions(threshold=sys.maxsize)

config_path = "../config/load_dbpedia.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# end with
pprint('=' * 20 + 'Configs' + '=' * 20)
pprint(config)

df_train = load_clean_data(config['train_out_path'])
featurized_train = get_tfidf(df_train)
save_featurized_data(featurized_train, config['train_tfidf_name'])

# df_test = load_clean_data(config['test_out_path'])
# featurized_test = get_tfidf(df_test)
# save_featurized_data(featurized_test, config['test_tfidf_name'])
