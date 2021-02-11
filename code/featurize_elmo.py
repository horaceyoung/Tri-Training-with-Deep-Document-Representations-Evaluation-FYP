import yaml
import pandas as pd
import logging
import sys
import os
import numpy as np
import tensorflow_hub as hub
import sentencepiece as spm
import tensorflow as tf
import nltk

tf.compat.v1.enable_eager_execution()
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize


def load_clean_data(path):
    data = pd.read_csv(path)
    pprint("Data from path {} loaded".format(path))
    return data


def get_elmo(df, batch_size):
    def _restric_len(sentence, max_seq_len=128):
        tokens = word_tokenize(sentence)
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        return " ".join(tokens)
        # end def

    pprint("Converting to ELMO....")
    max_seq_len = 64
    pprint(df["text"])
    df["text"] = df["text"].apply(lambda x: _restric_len(x, max_seq_len))
    pprint(df["text"])
    df_pooled_all = []
    # df_seq_all = []
    for df_chunk in np.array_split(df, batch_size):
        pooled_embeddings = ELMO_EMBED.signatures["default"](
            text=tf.convert_to_tensor(df_chunk["text"].values)
        )["default"]
        # seq_embeddings = ELMO_EMBED.signatures['default'](text=tf.convert_to_tensor(df_chunk['text'].values))['elmo']

        df_pooled_all.append(pooled_embeddings.numpy())
        # df_seq_all.append(seq_embeddings.numpy())
        # pprint(pooled_embeddings.shape)
        # pprint(len(df_pooled_all))

    df_pooled_all = np.concatenate(df_pooled_all, axis=0)
    # df_seq_all = np.concatenate(df_seq_all, axis=0)

    df_pooled = df.copy()
    df_seq = df.copy()
    df_pooled["text"] = list(df_pooled_all)
    # df_seq['text'] = list(df_seq_all)
    return df_pooled


def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)


logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

ELMO_EMBED = hub.load("https://tfhub.dev/google/elmo/3")
nltk.download("punkt")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path = "../config/load_yelp.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# end with
pprint("=" * 20 + "Configs" + "=" * 20)
pprint(config)

pprint("Featurizing starts--")
df_train = load_clean_data(config["train_out_path"])
pooled_train = get_elmo(df_train, 3250)
save_featurized_data(pooled_train, config["train_pool_elmo_name"])
# save_featurized_data(seq_train, config['train_seq_elmo_name'])

df_test = load_clean_data(config["test_out_path"])
pooled_test = get_elmo(df_test, 250)
save_featurized_data(pooled_test, config["test_pool_elmo_name"])
# save_featurized_data(seq_test, config['test_seq_elmo_name'])
