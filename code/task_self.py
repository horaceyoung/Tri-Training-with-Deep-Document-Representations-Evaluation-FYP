import numpy as np
import yaml
import logging
import sys
from models import *
from self_training_model import SelfTraining
from pprint import pprint


file = "../config/load_yelp.yaml"
with open(file, "r") as f:
    config = yaml.safe_load(f)
# end with
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

# Initialize the tri_training model
for ur in [0.5]:
    st_instance = SelfTraining(
        representations=["tfidf", "tfidf", "tfidf"],
        classifiers=[get_svm(), get_svm(), get_svm()],
        config=config,
        unlabel_rate=ur,
        iter=10,
    )
    st_instance.fit()
    st_instance.save()
