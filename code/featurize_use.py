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

tf.disable_v2_behavior()


def load_clean_data(path):
    data = pd.read_csv(path)
    pprint("Data from path {} loaded".format(path))
    return data


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)


def get_use(df):
    #### embed all documents with tfidf
    pprint('=' * 20 + 'Embedding with use' + '=' * 20)
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    encodings = USE_EMBED(
        inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))
    
    
    with tf.Session() as sess:
        spm_path = sess.run(USE_EMBED(signature="spm_path"))

    sp = spm.SentencePieceProcessor()
    with tf.io.gfile.GFile(spm_path, mode="rb") as f:
        sp.LoadFromSerializedProto(f.read())
    print("SentencePiece model loaded at {}.".format(spm_path))
    
    message_embeddings_all = []
    for df_chunk in np.array_split(df, 100):
        values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, df_chunk['text'])

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(
                encodings,
                feed_dict={input_placeholder.values: values,
                input_placeholder.indices: indices,
                input_placeholder.dense_shape: dense_shape})
        pprint(message_embeddings.shape)
        message_embeddings_all.append(message_embeddings)
        pprint(len(message_embeddings_all))
    
    message_embeddings_all = np.concatenate(message_embeddings_all, axis=0 )
    pprint(message_embeddings_all.shape)
    df['text'] = list(message_embeddings_all)
    return df

def save_featurized_data(df, path):
    logging.info("To pickle: started")
    df.to_pickle(path)


logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path = "../config/load_dbpedia.yaml"
USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
USE_EMBED = hub.Module(USE_MODULE_URL)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# end with
pprint('=' * 20 + 'Configs' + '=' * 20)
pprint(config)

pprint("Featurizing starts--")
df_train = load_clean_data(config['train_out_path'])
featurized_train = get_use(df_train)
save_featurized_data(featurized_train, config['train_use_name'])

# df_test = load_clean_data(config['test_out_path'])
# featurized_test = get_use(df_test)
# save_featurized_data(featurized_test, config['test_use_name'])
