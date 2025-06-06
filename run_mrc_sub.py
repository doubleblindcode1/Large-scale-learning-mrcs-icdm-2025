"""
    This experiment runs MRC-SUB on a dataset using a seed value that are given as command line argument.
"""

import warnings
from sklearn.model_selection import train_test_split
from MRCpy.phi import BasePhi, RandomFourierPhi
from MRCpy import MRC
from exp_utils import *

if __name__ == '__main__':

    warnings.simplefilter("ignore")

    # Get the command line arguments.
    dataset_name = sys.argv[1]
    rep_id = int(sys.argv[2])
    fit_intercept = True

    X, y, n, d, n_classes, dict_nnz = load_dataset_(dataset_name)

    # Initialization based on binary or multi class classification
    if n_classes == 2:
        one_hot = False
        max_iters = 150000
    elif n_classes > 2:
        one_hot = True
        max_iters = 15000
    else:
        raise ValueError('Invalid number of classes')

    i = rep_id

    # Training times
    train_time_mrc_subgrad = []

    # Classification error
    clf_error_mrc_subgrad = []

    # Upper bound
    upper_mrc_subgrad = []

    print("Repetition: ", i)

    # Save the common mrc parameters used for all
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, random_state=i, stratify=y)

    s = 0.01
    print('The regularization for MRCs is: ', s)
    print('Training size: ', X_train.shape)

    if d < 100:
        phi_ob = RandomFourierPhi(n_classes=n_classes,
                                  one_hot=one_hot,
                                  n_components=400,
                                  random_state=i,
                                  fit_intercept=fit_intercept).fit(X_train, y_train)

        X_train = phi_ob.transform(X_train)
        X_test = phi_ob.transform(X_test)

    # MRC subgradient using MRCpy
    init_time_mrc_subgrad = time.time()
    phi_ob = BasePhi(n_classes=n_classes,
                     one_hot=one_hot,
                     fit_intercept=fit_intercept).fit(X_train, y_train)

    print('\nTraining MRCs using MRC-SUB')
    tau_ = phi_ob.est_exp(X_train, y_train)
    lambda_ = s * (phi_ob.est_std(X_train, y_train))
    clf = MRC(loss='0-1',
              phi=phi_ob,
              s=s,
              solver='subgrad',
              max_iters=max_iters,
              one_hot=one_hot,
              fit_intercept=fit_intercept,
              deterministic=False)
    clf.minimax_risk(X_train, tau_, lambda_, n_classes)
    upper_mrc_subgrad.append(clf.get_upper_bound())
    train_time_mrc_subgrad.append(time.time() - init_time_mrc_subgrad)
    print('The worst-case error probability using MRC-SUB: ', upper_mrc_subgrad[-1])
    print('Training time using MRC-SUB: ', train_time_mrc_subgrad[-1])
    clf.deterministic = True
    clf_error_mrc_subgrad.append(clf.error(X_test, y_test))
    print('Classification error using MRC-SUB : ', clf_error_mrc_subgrad[-1])
    print('\n')

