import gc
import numpy as np
import tensorflow as tf
import yaml
import joblib
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import random
from tri_training_model import TriTraining
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# from utils.common import build_model, build_model_word_vector, build_conv_model, softmax

file = '../config/ttnews.yaml'
with open(file, 'r') as f:
    config = yaml.safe_load(f)
# end with

# Load labeled data, unlabeled data, and testing data
L = joblib.load(config['labeled_train_out'])
U = joblib.load(config['unlabeled_train_out'])
T = joblib.load(config['test_out'])
# Initialize the tri_training model
# Using the example of TFIDF and three RFC
tt_instance = TriTraining(representations=['tfidf', 'doc2vec', 'use'],
                          classifiers=[RandomForestClassifier(), LinearSVC(), AdaBoostClassifier()],
                          config=config)
tt_instance.fit(L, U)
print(tt_instance.score(T))

# To DO:
# 1. Logic to construct classifer models
