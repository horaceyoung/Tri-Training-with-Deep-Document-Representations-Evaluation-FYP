import numpy as np
import yaml
import logging
import sys
from models import *
from co_training_model import CoTraining
from pprint import pprint


file = "../config/load_dbpedia.yaml"
with open(file, "r") as f:
    config = yaml.safe_load(f)
# end with
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

# Initialize the co_training model
# Using the example of doc2vec and three RFC
for ur in [0.5]:
    co_instance = CoTraining(
        representations=["doc2vec", "doc2vec"],
        classifiers=[get_rfc(), get_rfc()],
        config=config,
        unlabel_rate=ur,
        iter=1,
    )
    co_instance.fit()
    co_instance.save()
