import yaml
import logging
import sys
import os
import pandas as pd
import numpy as np
from pprint import pprint
import torch
from random import randint
from models import InferSent


def load_clean_data(path):
    data = pd.read_csv(path)
    pprint("Data from path {} loaded".format(path))
    return data


def get_infersent(df, batch_size):
    pprint("Converting to infersent....")

    # df_pooled_all = []
    # for df_chunk in np.array_split(df, batch_size):
    texts = df["text"].values.tolist()
    print("texts with length {} created".format(len(texts)))
    pooled_output = model.encode(texts, bsize=128, tokenize=True, verbose=True)
    # df_pooled_all.append(pooled_output)
    # df_pooled_all = np.concatenate(df_pooled_all, axis=0)
    df_pooled = df.copy()

    df_pooled["text"] = pooled_output
    return df_pooled


def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)


# env settings
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)
print(os.getcwd())

# load inferdent model
model_version = 2
MODEL_PATH = "../code/InferSent/encoder/infersent%s.pkl" % model_version
params_model = {
    "bsize": 64,
    "word_emb_dim": 300,
    "enc_lstm_dim": 2048,
    "pool_type": "max",
    "dpout_model": 0.0,
    "version": model_version,
}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = (
    "../code/InferSent/GloVe/glove.840B.300d.txt"
    if model_version == 1
    else "../code/InferSent/fastText/crawl-300d-2M.vec"
)
model.set_w2v_path(W2V_PATH)
model.build_vocab_k_words(K=100000)


config_path = "../config/load_dbpedia.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# end with
pprint("=" * 20 + "Configs" + "=" * 20)
pprint(config)

pprint("Featurizing starts--")

df_train = load_clean_data(config["train_out_path"])
pooled_train = get_infersent(df_train.head(100), 560)
save_featurized_data(pooled_train, config["train_infersent_name"])

df_test = load_clean_data(config["test_out_path"])
pooled_test = get_infersent(df_test, 70)
save_featurized_data(pooled_test, config["test_infersent_name"])
