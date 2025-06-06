"""
    This experiment runs MRC-CCG and SVM-CCG on a dataset using a seed value that are given as command line argument.

"""

import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from MRCpy.phi import BasePhi, RandomFourierPhi
from exp_utils import *

from Libraries.L1SVM_CGCP.Our_methods import use_FOM_CGCP
from Libraries.L1SVM_CP.Our_methods import use_FOM_CP

from main_ccg_large_n.main import main_large_n
from main_ccg_large_n_multiclass.main import main_large_n_efficient_multiclass

from main_ccg_large_n_m.main import main_large_n_m
from main_ccg_large_n_m_sparse_binary.main import main_large_n_m_sparse_binary

import scipy as sp

if __name__ == '__main__':

    warnings.simplefilter("ignore")

    # Get the dataset name and rep_id from command line.
    dataset_name = sys.argv[1]
    rep_id = int(sys.argv[2])
    large_n = 1

    # Set the hyper-parameters
    eps_1 = 1e-2
    eps_2 = 1e-5
    n_max = 400
    k_max = 400

    max_iters = 200
    fit_intercept = True

    if dataset_name == "news20" or dataset_name == "rcv1" or dataset_name == "real_sim":
        is_sparse = True
    else:
        is_sparse = False

    X, y, n, d, n_classes, dict_nnz = load_dataset_(dataset_name)

    print('Unique classes: ', np.unique(y))
    if d > 2000:
        large_n = 0

    # Initialization based on binary or multi class classification
    if n_classes == 2:
        one_hot = False
    elif n_classes > 2:
        one_hot = True
    else:
        raise ValueError('Invalid number of classes')

    i = rep_id

    # Training times
    train_time_mrc_cg = []
    train_time_svm_cg = []

    # Classification error
    clf_error_mrc_cg = []
    clf_error_svm_cg = []

    # Upper bound
    upper_mrc_cg = []
    obj_svm_cg = []

    print('\n')
    print("Repetition: ", i)

    # Save the common mrc parameters used for all
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, random_state=i, stratify=y)

    if is_sparse:
        dict_nnz_X_train = {}
        for idx, row in enumerate(X_train):
            dict_nnz_X_train[idx] = row.nonzero()[1].tolist()

    s = 0.01
    print('\nThe regularization for MRC and SVM is: ', s)
    print('Training size: ', X_train.shape)

    if d < 100:
        phi_ob = RandomFourierPhi(n_classes=n_classes,
                                  one_hot=one_hot,
                                  n_components=400,
                                  random_state=i,
                                  fit_intercept=fit_intercept).fit(X_train, y_train)

        X_train = phi_ob.transform(X_train)
        X_test = phi_ob.transform(X_test)

    # MRC-CG
    print("Training MRCs using MRC-CCG")
    init_time_mrc_cg = time.time()
    if not is_sparse:
        phi_ob = BasePhi(n_classes=n_classes,
                         one_hot=one_hot,
                         fit_intercept=fit_intercept).fit(X_train, y_train)

    if large_n == 1 and n_classes == 2:
        print('\nLarge number of samples\n')
        mu, nu, R, R_k, mrc_cg_time, solver_times, init_mrc_cg_time = main_large_n(X_train,
                                                                                  y_train,
                                                                                  phi_ob,
                                                                                  s,
                                                                                  n_max,
                                                                                  max_iters,
                                                                                  eps_1)
        idx_cols = np.arange(phi_ob.len_)
    elif large_n == 1 and n_classes > 2:
        print('\nLarge number of samples and multiple classes \n')
        mu, nu, R, R_k, mrc_cg_time, solver_times, init_mrc_cg_time, constr_dict = main_large_n_efficient_multiclass(X_train,
                                                                                                                  y_train,
                                                                                                                  phi_ob,
                                                                                                                  s,
                                                                                                                  n_max,
                                                                                                                  max_iters,
                                                                                                                  eps_1)
        idx_cols = np.arange(phi_ob.len_)
        no_of_constraints = 0
        for x_i, subset_arr in constr_dict.items():
            for subset in subset_arr:
                no_of_constraints = no_of_constraints + 1
        print('###### The total number of constraints selected : ', no_of_constraints)
    elif is_sparse:
        print('\n Large number of samples and features binary classification (Sparse implementation)\n')
        mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, mrc_cg_time, init_mrc_cg_time, n_tries = main_large_n_m_sparse_binary(X_train,
                                                                                                                                         y_train,
                                                                                                                                         fit_intercept,
                                                                                                                                         s,
                                                                                                                                         n_max,
                                                                                                                                         k_max,
                                                                                                                                         eps_1,
                                                                                                                                         eps_2,
                                                                                                                                         is_sparse,
                                                                                                                                         dict_nnz_X_train,
                                                                                                                                         max_iters)
    else:
        print('\n Large number of samples and features \n')
        mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, mrc_cg_time, init_mrc_cg_time, n_tries = main_large_n_m(X_train,
                                                                                                                             y_train,
                                                                                                                             phi_ob,
                                                                                                                             s,
                                                                                                                             n_max,
                                                                                                                             k_max,
                                                                                                                             eps_1,
                                                                                                                             eps_2,
                                                                                                                             max_iters)
    print('\n')
    print('Training time MRC-CCG: ', time.time() - init_time_mrc_cg)
    print('The worst-case error probability using MRC-CCG: ', R)
    train_time_mrc_cg.append(time.time() - init_time_mrc_cg)
    upper_mrc_cg.append(R)
    if large_n == 1 and n_classes > 2:
        X_test_transform = np.hstack(([[1]] * X_test.shape[0], phi_ob.transform(X_test)))
        hy_x = predict_proba(X_test_transform, mu, nu, n_classes, large_n)
    elif is_sparse:
        X_transformed = sp.sparse.hstack([sp.sparse.csr_matrix([[1]] * X_test.shape[0]), X_test]).tocsr()
        hy_x = predict_proba_sparse(X_transformed[:, idx_cols], mu, nu, n_classes)
    else:
        hy_x = predict_proba(phi_ob.eval_x(X_test)[:, :, idx_cols], mu, nu, n_classes, large_n)
    clf_error_mrc_cg.append(np.average(y_test != np.argmax(hy_x, axis=1)))
    print('The classification error using MRC-CCG is: ', clf_error_mrc_cg[-1])
    print('\n')

    ######## SVM-CG
    print("Training SVMs using SVM-CCG")
    init_time_svm_cg = time.time()
    lam_max = np.max(np.sum(np.abs(X_train), axis=0))
    lam = s
    relative_lam = lam / lam_max

    if n_classes == 2:
       y_train1 = y_train.copy()
       y_train1[y_train1 != y_train[0]] = -1
       y_train1[y_train1 == y_train[0]] = 1
       if large_n == 1:
           obj, time_total, time_CG, beta, beta0 = use_FOM_CP(X_train, y_train1, lam, relative_lam, tol=eps_1)
           y_pred_svm_cg = np.sign(X_test @ beta + beta0)
       else:
            if is_sparse:
                obj, time_total, time_CG, beta, beta0 = use_FOM_CGCP(X_train, y_train1, lam, relative_lam, tol=eps_1, is_sparse=True, dict_nnz=dict_nnz_X_train)
                y_pred_svm_cg = np.sign(X_test.dot(beta) + beta0)
            else:
                obj, time_total, time_CG, beta, beta0 = use_FOM_CGCP(X_train, y_train1, lam, relative_lam, tol=eps_1)
                y_pred_svm_cg = np.sign(X_test @ beta + beta0)
       y_test1 = y_test.copy()
       y_test1[y_test1 != y_train[0]] = -1
       y_test1[y_test1 == y_train[0]] = 1
       clf_error_svm_cg.append(np.average(y_pred_svm_cg != y_test1))

       train_time_svm_cg.append(time.time() - init_time_svm_cg)
       obj_svm_cg.append(obj)
       # Compute the time and error
       print('Time taken by SVM-CCG: ', train_time_svm_cg[-1])
       print('The error using SVM-CCG is: ', clf_error_svm_cg[-1])
       print('\n')

    else:
       print('\n###### SVM-CCG is not available for mult-class setting.\n')

