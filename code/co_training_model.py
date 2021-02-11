import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
from pprint import pprint
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.DEBUG)


class CoTraining:
    def __init__(self, representations, classifiers, config, unlabel_rate, iter):
        """
        Args:
            representations: a list of strings of size 3 containing the representations used by three classifiers
                            in order, e.g.['TFIDF', 'Doc2Vec', 'BERT']
            classifiers: a list of pre-compiled classifier models of size 3
                        containing three classifiers in order,
                        e.g.[RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()]
            config: configuration file
            unlabel_rate: the percentage of unlabeled data in training set
            iter: iterations for fitting
        """
        self.representations = representations
        self.classifiers = classifiers
        self.config = config
        self.unlabel_rate = unlabel_rate
        self.initialized = False
        self.lb = preprocessing.LabelBinarizer()
        self.labeled_data = []
        self.labeled_class = []
        self.unlabeled_data = []
        self.labeled_test_data = []
        self.labeled_test_class = []
        self.dim = 0
        self.index1 = []
        self.index2 = []
        self.iter = iter
        np.random.seed(0)

    def load_representations(self):
        """
        Load training .pkl representations to the class object
        """

        for i in range(0, 2):
            # Loading training data from pickle files
            train_data = pd.read_pickle(
                self.config["train_path"] + self.representations[i] + ".pkl"
            )
            # Select a portion from training data, comment the line below to train on entire dataset
            # Train data is an array of arrays
            train_data = train_data.groupby(self.config["class_name"]).apply(
                lambda x: x.sample(frac=0.01, random_state=2021)
            )
            # Split training data into labeled set and unlabeled set
            (
                labeled_train,
                unlabeled_train,
                labeled_class,
                unlabeled_class,
            ) = train_test_split(
                train_data["text"].values,
                train_data[self.config["class_name"]].values,
                test_size=self.unlabel_rate,
                random_state=self.config["random_states"]["label_unlabel_split"],
            )

            labeled_train = np.stack(labeled_train)
            unlabeled_train = np.stack(unlabeled_train)
            labeled_class = np.stack(labeled_class)

            self.labeled_data.append(np.array(labeled_train))
            self.labeled_class.append(np.array(labeled_class))
            self.unlabeled_data.append(np.array(unlabeled_train))

        # Randomly select column indexes to be splitted in half
        self.dim = labeled_train[0].shape[0]
        column_index = list(range(0, self.dim))
        random.seed(10)
        random.shuffle(column_index)
        self.index1 = column_index[: int(self.dim / 2)]
        self.index2 = column_index[int(self.dim / 2) :]

        logging.debug("Labeled data has size {}".format(len(self.labeled_data[0])))
        logging.debug("Unlabeled data has size {}".format(len(self.unlabeled_data[0])))
        logging.debug("labeled data has size {}".format(len(self.labeled_class[0])))

    def load_test_representations(self):
        """
        Load training .pkl representations to the class object
        """

        for i in range(0, 2):
            # Load testing data from pickle files
            test_data = pd.read_pickle(
                self.config["test_path"] + self.representations[i] + ".pkl"
            )
            logging.info(
                "Testing data with embedding {} loaded".format(self.representations[i])
            )
            self.labeled_test_data.append(np.stack(test_data["text"].values))
            self.labeled_test_class.append(test_data[self.config["class_name"]])

    def fit(self):
        """
        Fitting two classifiers on training data
        """

        if not self.initialized:
            self.load_representations()
            self.initialized = True

        # Randomly sample 75 examples from unlabeled data to form the U_prime set
        U_prime = [
            self.unlabeled_data[0][
                np.random.randint(self.unlabeled_data[0].shape[0], size=75), :
            ],
            self.unlabeled_data[1][
                np.random.randint(self.unlabeled_data[1].shape[0], size=75), :
            ],
        ]

        for i in range(self.iter):
            for j in range(0, 2):
                # In each iteration, feature space of labeled data is partitioned in half to train each classifier
                if j == 0:
                    l_train_x_sampled = self.labeled_data[j][:, self.index1]
                else:
                    l_train_x_sampled = self.labeled_data[j][:, self.index2]
                l_train_y_sampled = self.labeled_class[j]

                # Train the classifier with sampled data
                self.classifiers[j].fit(l_train_x_sampled, l_train_y_sampled)

                # Make predictions on U_prime and add 4 most confident samples to L
                # For SVM classifier, use decision_function, otherwise use predict_proba
                if isinstance(self.classifiers[j], OneVsRestClassifier):
                    if j == 0:
                        pred_confidence = (
                            self.classifiers[j]
                            .decision_function(U_prime[j][:, self.index1])
                            .tolist()
                        )
                    else:
                        pred_confidence = (
                            self.classifiers[j]
                            .decision_function(U_prime[j][:, self.index2])
                            .tolist()
                        )

                    for k in range(0, len(pred_confidence)):
                        # Select the unlabeled samples with the highest confidence values
                        pred_confidence[k] = min(
                            [low_pos for low_pos in pred_confidence[k] if low_pos > 0],
                            default=100, # A large number, which is unlikely to be ranked first
                        )
                    sorted_index = sorted(
                        range(len(pred_confidence)), key=lambda k: pred_confidence[k]
                    )

                    # Add top 4 confident data to L
                    self.labeled_data[j] = np.append(
                        self.labeled_data[j], U_prime[j][sorted_index[0:4], :], axis=0
                    )

                    if j == 0:
                        pred_top4 = self.classifiers[j].predict(
                            U_prime[j][sorted_index[0:4], :][:, self.index1]
                        )
                    else:
                        pred_top4 = self.classifiers[j].predict(
                            U_prime[j][sorted_index[0:4], :][:, self.index2]
                        )

                    self.labeled_class[j] = np.append(
                        self.labeled_class[j], pred_top4, axis=0
                    )

                    # Remove added 4 data from U_prime
                    U_prime[j] = np.delete(U_prime[j], sorted_index[0:4], 0)
                    # Randomly replenish U_prime with 4 unlabeled data
                    U_prime[j] = np.append(
                        U_prime[j],
                        self.unlabeled_data[j][
                            np.random.randint(self.unlabeled_data[j].shape[0], size=4),
                            :,
                        ],
                        axis=0,
                    )
                elif (
                    isinstance(self.classifiers[j], RandomForestClassifier)
                    or isinstance(self.classifiers[j], GradientBoostingClassifier)
                    or isinstance(self.classifiers[j], GaussianNB)
                    or isinstance(self.classifiers[j], MLPClassifier)
                ):
                    if j == 0:
                        pred_confidence = (
                            self.classifiers[j]
                            .predict_proba(U_prime[j][:, self.index1])
                            .tolist()
                        )
                    else:
                        pred_confidence = (
                            self.classifiers[j]
                            .predict_proba(U_prime[j][:, self.index2])
                            .tolist()
                        )

                    for k in range(0, len(pred_confidence)):
                        # Select the unlabeled samples with the highest confidence values
                        pred_confidence[k] = max(pred_confidence[k], default=0)
                    sorted_index = sorted(
                        range(len(pred_confidence)), key=lambda k: pred_confidence[k]
                    )
                    # Add top 4 confident data in L
                    self.labeled_data[j] = np.append(
                        self.labeled_data[j], U_prime[j][sorted_index[0:4], :], axis=0
                    )
                    if j == 0:
                        pred_top4 = self.classifiers[j].predict(
                            U_prime[j][sorted_index[0:4], :][:, self.index1]
                        )
                    else:
                        pred_top4 = self.classifiers[j].predict(
                            U_prime[j][sorted_index[0:4], :][:, self.index2]
                        )
                    self.labeled_class[j] = np.append(
                        self.labeled_class[j], pred_top4, axis=0
                    )

                    # Remove added 4 data from U_prime
                    U_prime[j] = np.delete(U_prime[j], sorted_index[0:4], 0)
                    # Randomly replenish U_prime with 4 unlabeled data
                    U_prime[j] = np.append(
                        U_prime[j],
                        self.unlabeled_data[j][
                            np.random.randint(self.unlabeled_data[j].shape[0], size=4),
                            :,
                        ],
                        axis=0,
                    )

    def predict(self):
        """
        Make predictions based on the product of classification probabilities of the two classifiers
        """
        logging.info("Prediction started")
        pred1 = self.classifiers[0].predict(self.labeled_test_data[0][:, self.index1])
        pred2 = self.classifiers[1].predict(self.labeled_test_data[1][:, self.index2])

        if isinstance(self.classifiers[0], OneVsRestClassifier):
            return pred1
        else:
            proba1 = self.classifiers[0].predict_proba(
                self.labeled_test_data[0][:, self.index1]
            )
            proba2 = self.classifiers[1].predict_proba(
                self.labeled_test_data[1][:, self.index2]
            )
            combined_result = []
            for i in range(0, len(proba1)):
                products = []
                for num1, num2 in zip(proba1[i], proba2[i]):
                    products.append(num1 * num2)
                combined_result.append(products)

            for i in range(0, len(combined_result)):
                combined_result[i] = (
                    combined_result[i].index(max(combined_result[i])) + 1
                )
            return np.array(combined_result)

    def score(self):
        """
        Returns the prediction score of the tri-training instance
        """
        self.load_test_representations()
        pprint("Validation truth is :" + str(self.labeled_test_class[0]))
        return classification_report(self.labeled_test_class[0], self.predict())

    def save(self):
        """
        Create directory to store classifiers and training result
        """
        model_name = ""
        for i in range(0, 2):
            model_name += (
                str(self.classifiers[i].__class__.__name__)
                + "_"
                + self.representations[i]
                + "_"
            )
        directory = (
            self.config["result_path"]
            + "co/"
            + str(self.unlabel_rate)
            + "unlabeled_rate_"
            + model_name
        )
        print(directory)
        print(os.path.exists(directory))
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(0, 2):
            with open(
                directory + "/classifier_{}".format(str(i)) + ".pkl", "wb+"
            ) as clf:
                pickle.dump(self.classifiers[i], clf)

        with open(directory + "/results.txt", "w+") as rst:
            print(self.score())
            rst.write(self.score())
