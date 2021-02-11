import yaml
import pandas as pd
import logging
import sys
import os
import numpy as np
import tensorflow_hub as hub
import sentencepiece as spm
import tensorflow as tf
import tensorflow_text
import nltk
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize


def load_clean_data(path):
    data = pd.read_csv(path)
    pprint("Data from path {} loaded".format(path))
    return data


def get_bert(df, batch_size):
    pprint("Converting to bert....")
    df_pooled_all = []
    # df_seq_all = []
    texts = list(df["text"].values)
    count = 0
    for chunk in np.array_split(texts, batch_size):
        texts = preprocess(chunk)
        pooled_output = BERT_EMBED(texts)["pooled_output"]
        count += 1
        print("Chunk {} has completed".format(str(count)))
        df_pooled_all.append(pooled_output.numpy())
    # df_seq_all.append(seq_embeddings.numpy())

    df_pooled_all = np.concatenate(df_pooled_all, axis=0)
    # df_seq_all = np.concatenate(df_seq_all, axis=0)

    df_pooled = df.copy()
    # df_seq = df.copy()
    print(len(list(df_pooled_all)))
    df_pooled["text"] = list(df_pooled_all)
    # df_seq['text'] = list(df_seq_all)
    return df_pooled


def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)


logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

preprocess = hub.load("https://tfhub.dev/tensorflow/albert_en_preprocess/2")

BERT_EMBED = hub.load("https://tfhub.dev/tensorflow/albert_en_base/2")

config_path = "../config/load_yelp.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# end with
pprint("=" * 20 + "Configs" + "=" * 20)
pprint(config)

pprint("Featurizing starts--")

df_train = load_clean_data(config["train_out_path"])
pooled_train = get_bert(df_train, 6500)
save_featurized_data(pooled_train, config["train_pool_bert_name"])
# save_featurized_data(seq_train, config['train_seq_bert_name'])

"""df_test = load_clean_data(config['test_out_path'])
pooled_test = get_bert(df_test, 250)
save_featurized_data(pooled_test, config['test_pool_bert_name'])
# save_featurized_data(seq_test, config['test_seq_bert_name'])"""
