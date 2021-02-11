import logging
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.DEBUG)


class SelfTraining:
    def __init__(self, representations, classifiers, config, unlabel_rate, iter):
        """
        Args:
            representations: a list of strings of size 3 containing the representations used by three classifiers
                            in order, e.g.['TFIDF', 'Doc2Vec', 'BERT']
            classifiers: a list of pre-compiled classifier models of size 3
                        containing three classifiers in order,
                        e.g.[RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()]
            config: configuration file
            iter: iterations for fitting
        """
        self.representations = representations
        self.classifiers = classifiers
        self.config = config
        self.unlabel_rate = unlabel_rate
        self.initialized = False
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

        #
        for i in range(0, 3):
            if isinstance(self.classifiers[i], MLPClassifier):
                self.classifiers[i].set_params(
                    hidden_layer_sizes=(
                        int(
                            (
                                config["out_dim"]
                                + config["rep_dim"][self.representations[i]]
                            )
                            / 2
                        )
                    )
                )
                logging.info(
                    "Classifier {} is a MLPClassifier, hidden layer size has been set to {}".format(
                        i,
                        int(
                            (
                                config["out_dim"]
                                + config["rep_dim"][self.representations[i]]
                            )
                            / 2
                        ),
                    )
                )

    def load_representations(self):
        """
        Args:
            labeled_train: labeled training data
            unlabeled_train: unlabeled training data
        """
        for i in range(0, 3):
            # loading training data from pickle files
            train_data = pd.read_pickle(
                self.config["train_path"] + self.representations[i] + ".pkl"
            )
            # select a portion from training data, comment the line below to train on entire dataset
            # train data is an array of arrays
            train_data = train_data.groupby(self.config["class_name"]).apply(
                lambda x: x.sample(frac=0.01, random_state=2021)
            )
            # split training data into labeled set and unlabeled set
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
            # encode labeled_class to one-hot encoding
            # labeled_class = labeled_class
            unlabeled_class = np.stack(unlabeled_class)

            self.labeled_data.append(np.array(labeled_train))
            self.labeled_class.append(np.array(labeled_class))
            self.unlabeled_data.append(np.array(unlabeled_train))

        logging.debug("Labeled data has size {}".format(len(self.labeled_data[0])))
        logging.debug("Unlabeled data has size {}".format(len(self.unlabeled_data[0])))
        logging.debug("labeled data has size {}".format(len(self.labeled_class[0])))

    def load_test_representations(self):
        """
        Args:
            labeled_test: labeled testing data
        """
        for i in range(0, 3):
            # loading training data from pickle files
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
        Args:
            labeled_train: labeled training data
            unlabeled_train: unlabeled training data
        """
        # Train the classifiers from bootstrap sampling of labeled_train
        if not self.initialized:
            self.load_representations()
            self.initialized = True

        U_prime = [
            self.unlabeled_data[0][
                np.random.randint(self.unlabeled_data[0].shape[0], size=75), :
            ],
            self.unlabeled_data[1][
                np.random.randint(self.unlabeled_data[1].shape[0], size=75), :
            ],
            self.unlabeled_data[2][
                np.random.randint(self.unlabeled_data[2].shape[0], size=75), :
            ],
        ]

        for i in range(self.iter):
            for j in range(0, 3):
                # train each classifier with labeled data L
                self.classifiers[j].fit(self.labeled_data[j], self.labeled_class[j])

                # make predictions on U_prime and add 4 most confident samples to L
                if isinstance(self.classifiers[j], OneVsRestClassifier):
                    pred_confidence = (
                        self.classifiers[j].decision_function(U_prime[j]).tolist()
                    )
                    for k in range(0, len(pred_confidence)):
                        # select the unlabeled samples with the highest confidence values
                        pred_confidence[k] = min(
                            [low_pos for low_pos in pred_confidence[k] if low_pos > 0],
                            default=100,
                        )
                    sorted_index = sorted(
                        range(len(pred_confidence)), key=lambda k: pred_confidence[k]
                    )

                    # add top 4 confident data in L
                    self.labeled_data[j] = np.append(
                        self.labeled_data[j], U_prime[j][sorted_index[0:4], :], axis=0
                    )
                    self.labeled_class[j] = np.append(
                        self.labeled_class[j],
                        self.classifiers[j].predict(U_prime[j][sorted_index[0:4], :]),
                        axis=0,
                    )

                    # remove added 4 data from U_prime
                    U_prime[j] = np.delete(U_prime[j], sorted_index[0:4], 0)
                    # randomly replenish U_prime with 4 unlabeled data
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
                    pred_confidence = (
                        self.classifiers[j].predict_proba(U_prime[j]).tolist()
                    )
                    for k in range(0, len(pred_confidence)):
                        # select the unlabeled samples with the highest confidence values
                        pred_confidence[k] = max(pred_confidence[k], default=0)
                    sorted_index = sorted(
                        range(len(pred_confidence)), key=lambda k: pred_confidence[k]
                    )
                    print(sorted_index[0:4])
                    # add top 4 confident data in L
                    self.labeled_data[j] = np.append(
                        self.labeled_data[j], U_prime[j][sorted_index[0:4], :], axis=0
                    )
                    self.labeled_class[j] = np.append(
                        self.labeled_class[j],
                        self.classifiers[j].predict(U_prime[j][sorted_index[0:4], :]),
                        axis=0,
                    )

                    # remove added 4 data from U_prime
                    U_prime[j] = np.delete(U_prime[j], sorted_index[0:4], 0)
                    # randomly replenish U_prime with 4 unlabeled data
                    U_prime[j] = np.append(
                        U_prime[j],
                        self.unlabeled_data[j][
                            np.random.randint(self.unlabeled_data[j].shape[0], size=4),
                            :,
                        ],
                        axis=0,
                    )

    def predict(self):
        logging.info("Prediction started")
        pred = np.asarray(
            [
                self.classifiers[i].predict(self.labeled_test_data[i])
                for i in range(0, 3)
            ]
        )
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        # pprint("Validation prediction is :" + str(pred[0]))
        return pred[0]

    def score(self):
        self.load_test_representations()
        # pprint("Validation truth is :" + str(self.labeled_test_class[0]))
        return classification_report(self.labeled_test_class[0], self.predict())

    def save(self):
        # create directory to store classifiers and training result
        model_name = ""
        for i in range(0, 3):
            model_name += (
                str(self.classifiers[i].__class__.__name__)
                + "_"
                + self.representations[i]
                + "_"
            )
        directory = (
            self.config["result_path"]
            + "self/"
            + str(self.unlabel_rate)
            + "unlabeled_rate_"
            + model_name
        )
        print(directory)
        print(os.path.exists(directory))
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(0, 3):
            with open(
                directory + "/classifier_{}".format(str(i)) + ".pkl", "wb+"
            ) as clf:
                pickle.dump(self.classifiers[i], clf)

        with open(directory + "/results.txt", "w+") as rst:
            print(self.score())
            rst.write(self.score())
