import numpy as np
import yaml
import sys
import logging
from models import *
from tri_training_model import TriTraining
from pprint import pprint


file = "../config/load_dbpedia.yaml"
with open(file, "r") as f:
    config = yaml.safe_load(f)
# end with
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=100)

# Initialize the tri_training model
# Using the example of doc2vec and three RFC
tt_instance = TriTraining(
    representations=["doc2vec", "doc2vec", "doc2vec"],
    classifiers=[get_rfc(), get_rfc(), get_rfc()],
    config=config,
    unlabel_rate=0.2,
)
tt_instance.fit()
tt_instance.save()
