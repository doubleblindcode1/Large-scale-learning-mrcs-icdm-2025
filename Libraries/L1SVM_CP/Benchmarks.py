
import cvxpy as cp

from sklearn.linear_model import SGDClassifier
from smoothing_hinge_loss import *
import time

import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"




def use_SCS(X_train, y_train, lam):
    N,P = np.shape(X_train)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    reg = cp.norm(beta, 1)
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    st = time.time()
    SCS_obj = prob.solve(solver = 'SCS', verbose=True)
    ed = time.time()
    SCS_time = ed - st
    
    beta_ = np.reshape(beta.value, (P,))
    beta0_ = beta0.value
    constraints = np.ones(N) - y_train * (np.dot(X_train, beta_) + beta0_) 
    SCS_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_))
    
    
    return SCS_obj, SCS_time, beta_, beta0_





def use_SGD(X_train, y_train, lam, max_iter):
	N = X_train.shape[0]

	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
	clf = SGDClassifier(loss="hinge",
						penalty="l1",
						alpha=lam / float(N),
						fit_intercept=True,
						max_iter=max_iter,
						tol=None,
						learning_rate='optimal')

	start   = time.time()
	clf.fit(X_train, y_train)
	SGD_time = time.time() - start

	beta_SGD = clf.coef_[0]
	b0_SGD = clf.intercept_[0]
	constraints = np.ones(N) - y_train * (np.dot(X_train, beta_SGD) + b0_SGD)
	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

	return SGD_obj, SGD_time, beta_SGD, b0_SGD







# def use_SGD(X_train, y_train, lam):
# 	N = X_train.shape[0]

# 	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 	clf = SGDClassifier(loss="hinge", 
# 						penalty="l1", 
# 						alpha=lam / float(N), 
# 						fit_intercept=True,
# 						max_iter=1e4,
# 						tol=None,
# 						learning_rate='optimal')

# 	start   = time.time()
# 	clf.fit(X_train, y_train)
# 	SGD_time = time.time() - start

# 	beta_SGD = clf.coef_[0]
# 	b0_SGD = clf.intercept_[0]
# 	constraints = np.ones(N) - y_train * (np.dot(X_train, beta_SGD) + b0_SGD) 
# 	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

# 	return SGD_obj, SGD_time, beta_SGD, b0_SGD


def use_FOM(X_train, y_train, lam, tau, max_iter):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    n_iter  = 1e5
    n_loop  = 1
    _, _, FOM_time, beta, beta0 = loop_smoothing_hinge_loss('hinge', 
                                                      'l1', 
                                                      X_train, 
                                                      y_train, 
                                                      lam, 
                                                      tau, 
                                                      n_loop, 
                                                      n_iter, 
                                                      f, 
                                                      is_sparse=False)
    beta0 = beta0/np.sqrt(N)
    cons = np.ones(N) - y_train*( np.dot(X_train, beta) + beta0) 
    FOM_obj = np.sum([max(cons[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta))
    
    return FOM_obj, FOM_time, beta, beta0





def use_Gurobi(X_train, y_train, lam):
    N,P = np.shape(X_train)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    reg = cp.norm(beta, 1)
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    st = time.time()
    gurobi_obj = prob.solve(solver = 'GUROBI', verbose=True)
    ed = time.time()
    gurobi_time = ed - st
    
    return gurobi_obj, gurobi_time, np.reshape(beta.value, (P,)), beta0.value


