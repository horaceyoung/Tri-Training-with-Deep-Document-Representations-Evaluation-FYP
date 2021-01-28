import gc
import numpy as np
import sklearn
import logging
import tensorflow as tf
import yaml
import joblib
import random
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from pprint import pprint

from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

class TriTraining:
    def __init__(self, representations, classifiers, config):
        """
        Args:
            representations: a list of strings of size 3 containing the representations used by three classifiers
                            in order, e.g.['TFIDF', 'Doc2Vec', 'BERT']
            classifiers: a list of pre-compiled classifier models of size 3
                        containing three classifiers in order,
                        e.g.[RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()]
            config: configuration file
        """
        self.representations = representations
        self.classifiers = classifiers
        self.config = config
        self.initialized = False # a flag indicating if data has been loaded
        self.labeled_data = []
        self.labeled_class = []
        self.unlabeled_data = []
        self.labeled_test_data = []
        self.labeled_test_class = []
        np.random.seed(0)

    def load_representations(self, labeled_train, unlabeled_train):
        """
        Args:
            labeled_train: labeled training data
            unlabeled_train: unlabeled training data
        """
        for i in range(0, 3):
            self.labeled_data.append(np.array([item[self.representations[i]] for item in labeled_train]))
            self.labeled_class.append(np.array([item['cat_en'] for item in labeled_train]))
            self.unlabeled_data.append(np.array([item[self.representations[i]] for item in unlabeled_train]))

    def load_test_representations(self, labeled_test):
        """
        Args:
            labeled_test: labeled testing data
        """
        for i in range(0, 3):
            self.labeled_test_data.append(np.array([item[self.representations[i]] for item in labeled_test]))
            self.labeled_test_class.append(np.array([item['cat_en'] for item in labeled_test]))

    def measure_error(self, j, k):
        """
        Measure the combined classification error of classifier j and k by the following equation:
        The number of misclassified examples which j and k agree upon / total examples j and k agree upon
        """
        j_prediction = self.classifiers[j].predict(self.labeled_data[j])
        k_prediction = self.classifiers[k].predict(self.labeled_data[k])
        wrong_index = np.logical_and(j_prediction != self.labeled_class[j], k_prediction == j_prediction)
        combined_error = sum(wrong_index) / sum(j_prediction == k_prediction)
        # pprint(j_prediction)
        # pprint(k_prediction)
        # pprint(wrong_index)
        # pprint(sum(k_prediction==j_prediction))
        # pprint(sum(wrong_index))
        return combined_error

    def fit(self, labeled_train, unlabeled_train):
        """
        Args:
            labeled_train: labeled training data
            unlabeled_train: unlabeled training data
        """
        # Train the classifiers from bootstrap sampling of labeled_train
        for i in range(0, 3):
            if not self.initialized:
                self.load_representations(labeled_train, unlabeled_train)
                self.initialized = True
            # debug: pprint(l_train_x.shape)
            # debug: pprint(l_train_y.shape)

            # Obtain the sampled training data
            l_train_x_sampled, l_train_y_sampled = \
                sklearn.utils.resample(np.copy(self.labeled_data[i]),  # deep copy class attributes
                                       np.copy(self.labeled_class[i]),
                                       random_state=self.config['bootstrap_sampling_random_state'])
            # Train the classifier with sampled data
            # debug: pprint(self.labeled_data[i])
            self.classifiers[i].fit(l_train_x_sampled, l_train_y_sampled)
            # logging.debug(self.classifiers[i].predict([self.labeled_data[i][0]]))

        e_prime = [0.5] * 3     # e denotes the upper bound of classification error rate
                                # with prime indicating the stored classification error rate of last round
        l_prime = [0] * 3       # l_prime denotes the number of labeled examples labeled by the other two classifiers from last round
        e = [0] * 3             # e denotes the upper bound of classification error rate of current round
        update = [False] * 3    # update is a flag indicating if the respective classifier has been updated
        improve = True          # to determine whether to stop tri-training, will be true if either of updates are true
        train_iter = 0
        labeled_data_i, labeled_class_i = [[]] * 3, [[]] * 3
        while improve:  # repeat until none of the classifiers are improved
            train_iter = train_iter + 1
            logging.debug("=" * 10 + " iteration {} has begun ".format(train_iter) + "=" * 10)

            # the following for loop will add unlabeled data agreed by classifier j,k to labeled data of classifier i
            for i in range(0, 3):
                j, k = np.delete(np.array([0, 1, 2]), i)  # remove the index of the current classifier
                update[i] = False # initialize update to be false
                e[i] = self.measure_error(j, k)

                logging.debug("classifier {}: combined error of classifier {} and {} is {}".format(i, j, k, e[i]))

                if e[i] < e_prime[i]:
                    j_prediction = self.classifiers[j].predict(self.unlabeled_data[j])
                    k_prediction = self.classifiers[k].predict(self.unlabeled_data[k])
                    # logging.debug(type(j_prediction))
                    # logging.debug(k_prediction)

                    # save the data where j and k agrees
                    labeled_data_i[i] = self.unlabeled_data[i][j_prediction == k_prediction]
                    # essentially j_prediction is equivalent to k prediction
                    labeled_class_i[i] = j_prediction[j_prediction == k_prediction]
                    logging.debug("Number of added labeled examples: {}".format(len(labeled_data_i[i])))
                else:
                    logging.info("e_prime{} > e{}".format(i, i))

                if l_prime[i] == 0: # classifier i has not been updated
                    l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    logging.info('l_prime {} is 0, new value {} has been assigned to l_prime{}'.format(i, l_prime[i], i))

                if l_prime[i] < len(labeled_class_i[i]):
                    logging.info("l_prime is less than L{}".format(i))
                    if e[i] * len(labeled_class_i[i]) < (e_prime[i] * l_prime[i]):
                        logging.info("e{} * L{} < e{}_prime * l{}_prime is satisfied".format(i, i, i, i))
                        update[i] = True
                        logging.info("h{} will be updated".format(i))
                    elif l_prime[i] > e[i]/(e_prime[i] - e[i]):
                        # randomly subsample Li
                        subsampled_index = np.random.choice(len(labeled_class_i[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                        labeled_data_i[i] = labeled_data_i[i][subsampled_index]
                        labeled_class_i[i] = labeled_class_i[i][subsampled_index]
                        logging.info('L{} has been subsampled to size {}'.format(i, len(subsampled_index)))
                        update[i] = True
                else:
                    logging.info("l_prime{} > L{}".format(i, i))

            for i in range(0, 3):
                if update[i]:
                    combined_labeled_data_i = np.append(self.labeled_data[i], labeled_data_i[i], axis=0)
                    combined_labeled_class_i = np.append(self.labeled_class[i], labeled_class_i[i], axis=0)
                    self.classifiers[i].fit(combined_labeled_data_i, combined_labeled_class_i)
                    e_prime[i] = e[i]
                    l_prime[i] = len(labeled_class_i[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement

    def predict(self, labeled_test):
        logging.info("Prediction started")
        pred = np.asarray([self.classifiers[i].predict(self.labeled_test_data[i]) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self, labeled_test):
        self.load_test_representations(labeled_test)
        return classification_report(self.labeled_test_class[0], self.predict(labeled_test))
