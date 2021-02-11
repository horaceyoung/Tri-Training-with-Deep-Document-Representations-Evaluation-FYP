import logging
import numpy as np
import os
import pandas as pd
import pickle
import sklearn

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pprint import pprint


logging.basicConfig(level=logging.DEBUG)


class TriTraining:
    def __init__(self, representations, classifiers, config, unlabel_rate):
        """
        Args:
            representations: a list of strings of size 3 containing the representations used by three classifiers
                            in order, e.g.['TFIDF', 'Doc2Vec', 'BERT']
            classifiers: a list of pre-compiled classifier models of size 3
                        containing three classifiers in order,
                        e.g.[RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()]
            config: configuration file
            unlabel_rate: the percentage of unlabeled data in training set
        """
        self.representations = representations
        self.classifiers = classifiers
        self.config = config
        self.unlabel_rate = unlabel_rate
        self.initialized = False  # a flag indicating if data has been loaded
        self.labeled_data = []
        self.labeled_class = []
        self.unlabeled_data = []
        self.labeled_test_data = []
        self.labeled_test_class = []
        np.random.seed(0)

    def load_representations(self):
        """
        Load training .pkl representations to the class object
        """

        for i in range(0, 3):
            # Loading training data from pickle files
            train_data = pd.read_pickle(
                self.config["train_path"] + self.representations[i] + ".pkl"
            )
            # Select 10% from training data, comment the line below to train on entire dataset
            train_data = train_data.sample(frac=0.1, random_state=2021)
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

            # Converting loaded array of arrays to a single 2-D numpy array
            labeled_train = np.stack(labeled_train)
            unlabeled_train = np.stack(unlabeled_train)
            labeled_class = np.stack(labeled_class)

            self.labeled_data.append(labeled_train)
            self.labeled_class.append(labeled_class)
            self.unlabeled_data.append(unlabeled_train)

        logging.debug("Labeled data has size {}".format(len(self.labeled_data[0])))
        logging.debug("Unlabeled data has size {}".format(len(self.unlabeled_data[0])))
        logging.debug("labeled data has size {}".format(len(self.labeled_class[0])))

    def load_test_representations(self):
        """
        Load training .pkl representations to the class object
        """
        for i in range(0, 3):
            # Load testing data from pickle files
            test_data = pd.read_pickle(
                self.config["test_path"] + self.representations[i] + ".pkl"
            )
            logging.info(
                "Testing data with embedding {} loaded".format(self.representations[i])
            )
            self.labeled_test_data.append(np.stack(test_data["text"].values))
            self.labeled_test_class.append(test_data[self.config["class_name"]])

    def measure_error(self, j, k):
        """
        Measure the combined classification error of classifier j and k by the following equation:
        The number of misclassified examples which j and k agree upon / total examples j and k agree upon
        """
        j_prediction = self.classifiers[j].predict(self.labeled_data[j])
        k_prediction = self.classifiers[k].predict(self.labeled_data[k])
        wrong_index = np.logical_and(
            j_prediction != self.labeled_class[j], k_prediction == j_prediction
        )
        combined_error = sum(wrong_index) / sum(j_prediction == k_prediction)
        pprint("Pridiction of classifier {}: ".format(str(j)) + str(j_prediction))
        pprint("Pridiction of classifier {}: ".format(str(k)) + str(k_prediction))
        pprint(
            "Number of predicted examples where classifier {} agrees with classifier {}: ".format(
                str(j), str(k)
            )
            + str(sum(k_prediction == j_prediction))
        )
        pprint(
            "Number of wrongly predicted labels agreed by two classifiers: "
            + str(sum(wrong_index))
        )
        return combined_error

    def fit(self):
        """
        Fitting the three classifiers on training data
        """
        # Train the classifiers from bootstrap sampling of labeled_train
        for i in range(0, 3):
            if not self.initialized:
                self.load_representations()
                self.initialized = True

            # Obtain the sampled training data
            l_train_x_sampled, l_train_y_sampled = sklearn.utils.resample(
                np.copy(self.labeled_data[i]),  # deep copy class attributes
                np.copy(self.labeled_class[i]),
                random_state=self.config["random_states"]["bootstrap_" + str(i)],
            )
            # Train the classifier with sampled data
            self.classifiers[i].fit(l_train_x_sampled, l_train_y_sampled)

        e_prime = [1] * 3  # e denotes the upper bound of classification error rate
        # with prime indicating the stored classification error rate of last round
        l_prime = [
            0
        ] * 3  # l_prime denotes the number of labeled examples labeled by the other two classifiers from last round
        e = [
            0
        ] * 3  # e denotes the upper bound of classification error rate of current round
        update = [
            False
        ] * 3  # update is a flag indicating if the respective classifier has been updated
        improve = True  # To determine whether to stop tri-training, will be true if either of updates are true
        train_iter = 0
        labeled_data_i, labeled_class_i = [[]] * 3, [[]] * 3
        while improve:  # Repeat until none of the classifiers are improved
            train_iter = train_iter + 1
            logging.debug(
                "=" * 10 + " Iteration {} has begun ".format(train_iter) + "=" * 10
            )

            # The following for loop will add unlabeled data agreed by classifier j,k to labeled data of classifier i
            for i in range(0, 3):
                j, k = np.delete(
                    np.array([0, 1, 2]), i
                )  # Remove the index of the current classifier
                update[i] = False  # Initialize update to be false
                e[i] = self.measure_error(j, k)

                logging.info(
                    "Classifier {}: combined error of classifier {} and {} is {}".format(
                        i, j, k, e[i]
                    )
                )

                if e[i] < e_prime[i]:
                    j_prediction = self.classifiers[j].predict(self.unlabeled_data[j])
                    k_prediction = self.classifiers[k].predict(self.unlabeled_data[k])

                    # Save the examples where j and k agrees
                    labeled_data_i[i] = self.unlabeled_data[i][
                        j_prediction == k_prediction
                    ]
                    # Essentially j_prediction is equivalent to k prediction
                    labeled_class_i[i] = j_prediction[j_prediction == k_prediction]
                    logging.info(
                        "Number of added labeled examples: {}".format(
                            len(labeled_data_i[i])
                        )
                    )
                else:
                    logging.info("e_prime{} > e{}".format(i, i))

                if l_prime[i] == 0:  # classifier i has not been updated
                    l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    logging.info(
                        "l_prime {} is 0, new value {} has been assigned to l_prime{}".format(
                            i, l_prime[i], i
                        )
                    )

                if l_prime[i] < len(labeled_class_i[i]):
                    logging.info("l_prime is less than L{}".format(i))
                    if e[i] * len(labeled_class_i[i]) < (e_prime[i] * l_prime[i]):
                        logging.info(
                            "e{} * L{} < e{}_prime * l{}_prime is satisfied".format(
                                i, i, i, i
                            )
                        )
                        update[i] = True
                        logging.info("Classifier {} will be updated".format(i))
                    elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                        # Randomly subsample Li
                        subsampled_index = np.random.choice(
                            len(labeled_class_i[i]),
                            int(e_prime[i] * l_prime[i] / e[i] - 1),
                        )
                        labeled_data_i[i] = labeled_data_i[i][subsampled_index]
                        labeled_class_i[i] = labeled_class_i[i][subsampled_index]
                        logging.info(
                            "L{} has been subsampled to size {}".format(
                                i, len(subsampled_index)
                            )
                        )
                        update[i] = True
                else:
                    logging.info("l_prime{} > L{}".format(i, i))

            # Update classifiers which can be improved
            for i in range(0, 3):
                if update[i]:
                    combined_labeled_data_i = np.append(
                        self.labeled_data[i], labeled_data_i[i], axis=0
                    )
                    combined_labeled_class_i = np.append(
                        self.labeled_class[i], labeled_class_i[i], axis=0
                    )
                    self.classifiers[i].fit(
                        combined_labeled_data_i, combined_labeled_class_i
                    )
                    e_prime[i] = e[i]
                    l_prime[i] = len(labeled_class_i[i])

            if update == [False] * 3:
                improve = False  # If no classifier was updated, no improvement

    def predict(self):
        """
        Returns the predicted class, which is acquired by majority vote between 3 classifiers
        """
        logging.info("Prediction started")
        pred = np.asarray(
            [
                self.classifiers[i].predict(self.labeled_test_data[i])
                for i in range(0, 3)
            ]
        )
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self):
        """
        Returns the prediction score of the tri-training instance
        """
        self.load_test_representations()
        return classification_report(self.labeled_test_class[0], self.predict(), digits=3)

    def save(self):
        """
        Create directory to store classifiers and training result
        """
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
            predict_score = self.score()
            rst.write(predict_score)
