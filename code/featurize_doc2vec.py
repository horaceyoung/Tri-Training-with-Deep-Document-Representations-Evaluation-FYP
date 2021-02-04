import yaml
import pandas as pd
import logging
import sys
import os
import numpy as np
import tensorflow_hub as hub
import sentencepiece as spm
import tensorflow.compat.v1 as tf
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec

tf.disable_v2_behavior()


def load_clean_data(path):
    data = pd.read_csv(path)
    pprint("Data from path {} loaded".format(path))
    return data


def get_doc2vec(df):
    # embed all documents with doc2vec
    df['text'] = list(np.array([model.infer_vector(doc.strip().split()) for doc in df['text'].values]))
    return df


def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)


logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

model = Doc2Vec.load("../resources/enwiki_dbow/doc2vec.bin")

config_path = "../config/load_dbpedia.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# end with
pprint('=' * 20 + 'Configs' + '=' * 20)
pprint(config)

pprint("Featurizing starts--")
df_train = load_clean_data(config['train_out_path'])
featurized_train = get_doc2vec(df_train)
save_featurized_data(featurized_train, config['train_doc2vec_name'])

# df_test = load_clean_data(config['test_out_path'])
# featurized_test = get_doc2vec(df_test)
# save_featurized_data(featurized_test, config['test_doc2vec_name'])
