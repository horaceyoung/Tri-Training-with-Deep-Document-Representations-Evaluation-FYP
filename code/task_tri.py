import numpy as np
import yaml
import logging
import sys
from models import *
from tri_training_model import TriTraining
from pprint import pprint


file = "../config/load_dbpedia.yaml"
with open(file, "r") as f:
    config = yaml.safe_load(f)
# end with
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=40)

# Initialize the tri_training model
# Using the example of doc2vec and three RFC
for ur in [0.5]:
    tt_instance = TriTraining(
        representations=["doc2vec", "doc2vec", "doc2vec"],
        classifiers=[get_mlp(), get_mlp(), get_mlp()],
        config=config,
        unlabel_rate=ur,
    )
    tt_instance.fit()
    tt_instance.save()
