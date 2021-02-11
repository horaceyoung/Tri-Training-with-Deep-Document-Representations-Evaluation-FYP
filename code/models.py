from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import tensorflow as tf


def get_rfc():
    rfc_classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=100,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=None,  # no. of features
        max_leaf_nodes=None,
        min_impurity_decrease=0.001,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=1,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )
    return rfc_classifier


def get_svm():
    linear_svc = OneVsRestClassifier(
        LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            tol=1e-5,
            C=1.0,
            multi_class="ovr",
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=2021,
            max_iter=2000,
            verbose=1,
        )
    )
    return linear_svc


def get_xgb():
    xgboost = GradientBoostingClassifier(
        loss="deviance",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=2021,
        max_features=None,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        verbose=1,
    )
    return xgboost


def get_gnb():
    gnb = GaussianNB(var_smoothing=1e-9)
    return gnb


def get_mlp():
    mlp = MLPClassifier(
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=2021,
        tol=1e-5,
        verbose=1,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    )
    return mlp
