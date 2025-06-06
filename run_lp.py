"""
    This experiment runs MRC-LP and SVM-LP on a dataset using a seed value that are given as command line argument.

"""

import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from MRCpy.phi import BasePhi, RandomFourierPhi
from MRCpy.datasets import *
from MRCpy.datasets import *
from exp_utils import *

from Libraries.L1SVM_CGCP.Benchmarks import use_Gurobi

if __name__ == '__main__':

    # Get the command line arguments.
    warnings.simplefilter("ignore")
    dataset_name = sys.argv[1]
    rep_id = int(sys.argv[2])

    fit_intercept = True

    X, y, n, d, n_classes, dict_nnz = load_dataset_(dataset_name)

    # Initialization based on binary or multi class classification
    if n_classes == 2:
        one_hot = False
    elif n_classes > 2:
        one_hot = True
    else:
        raise ValueError('Invalid number of classes')

    i = rep_id

    # Training times
    train_time_mrc_gurobi = []
    train_time_svm_gurobi = []

    # Classification error
    clf_error_mrc_gurobi = []
    clf_error_svm_gurobi = []

    # Upper bound
    upper_mrc_cg = []
    upper_mrc_gurobi = []
    obj_svm_gurobi = []

    print('\n')
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

    # MRC GUROBI
    print('Training MRC-LP using gurobipy')
    init_time_mrc_lp_gurobi = time.time()
    phi_ob = BasePhi(n_classes=n_classes,
                     one_hot=one_hot,
                     fit_intercept=fit_intercept).fit(X_train, y_train)
    tau_ = phi_ob.est_exp(X_train, y_train)
    lambda_ = s * (phi_ob.est_std(X_train, y_train))
    upper_mrc_gurobi_i, clf_error_mrc_gurobi_i, train_time_mrc_gurobi_i, solver_times_gurobi, R_k_gurobi = gurobi_lpsolve_mrc_primal(X_train,
                                                                                                                                    X_test,
                                                                                                                                    y_test,
                                                                                                                                    phi_ob,
                                                                                                                                    tau_,
                                                                                                                                    lambda_)
    print('Time taken by MRC-LP: ', train_time_mrc_gurobi_i)
    print('The worst-case error probability using MRC-LP: ', upper_mrc_gurobi_i)
    print('The classification error using MRC-LP: ', clf_error_mrc_gurobi_i)
    train_time_mrc_gurobi.append(train_time_mrc_gurobi_i)
    upper_mrc_gurobi.append(upper_mrc_gurobi_i)
    clf_error_mrc_gurobi.append(clf_error_mrc_gurobi_i)
    print('\n')

    # SVM-GUROBI
    print('Training SVM-LP using gurobipy')
    init_time_svm_gurobi = time.time()
    if n_classes == 2:
        y_train1 = y_train.copy()
        y_train1[y_train1 != y_train[0]] = -1
        y_train1[y_train1 == y_train[0]] = 1
        gurobi_obj, gurobi_time, beta, beta0 = use_Gurobi(X_train, y_train1, s)   
        y_pred_svm_gurobi = np.sign(X_test @ beta + beta0)
        y_test1 = y_test.copy()
        y_test1[y_test1 != y_train[0]] = -1
        y_test1[y_test1 == y_train[0]] = 1
        clf_error_svm_gurobi.append(np.average(y_pred_svm_gurobi != y_test1))
    else:
        ########## Multiclass L1-SVM gurobi
        init_time_l1_msvm = time.time()
        beta, beta0, gurobi_obj = l1_msvm_gurobi_lp(X_train, y_train, 100 / s)
        y_pred_svm_gurobi = np.argmax(X_test @ beta.T + beta0, axis=1)
        clf_error_svm_gurobi.append(np.average(y_pred_svm_gurobi != y_test))

    train_time_svm_gurobi.append(time.time() - init_time_svm_gurobi)
    obj_svm_gurobi.append(gurobi_obj)

    # Compute the time and error
    print('Time taken by SVM-LP: ', train_time_svm_gurobi[-1])
    print('The error using SVM-LP: ', clf_error_svm_gurobi[-1])
    print('\n')