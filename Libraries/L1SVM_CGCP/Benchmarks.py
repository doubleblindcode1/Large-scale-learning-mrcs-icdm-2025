
import cvxpy as cp
from sklearn.linear_model import SGDClassifier
from .smoothing_hinge_loss import *
import gurobipy as gp
import time
from gurobipy import GRB

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
    constraints = np.ones(N) - y_train * (X_train.dot(beta_) + beta0_) 
    SCS_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_))
    
    
    return SCS_obj, SCS_time, beta_, beta0_


# def use_SGD(X_train, y_train, lam):
# 	N = X_train.shape[0]

# 	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 	clf = SGDClassifier(loss="hinge", 
# 						penalty="l1", 
# 						alpha=lam / float(N), 
# 						fit_intercept=True,
# 						max_iter=1e5,
# 						tol=None,
# 						learning_rate='optimal')

# 	start   = time.time()
# 	clf.fit(X_train, y_train)
# 	SGD_time = time.time() - start

# 	beta_SGD = clf.coef_[0]
# 	b0_SGD = clf.intercept_[0]
# 	constraints = np.ones(N) - y_train * (X_train.dot(beta_SGD) + b0_SGD) 
# 	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

# 	return SGD_obj, SGD_time, beta_SGD, b0_SGD



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
	constraints = np.ones(N) - y_train * (X_train.dot(beta_SGD) + b0_SGD) 
	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

	return SGD_obj, SGD_time, beta_SGD, b0_SGD





def use_FOM(X_train, y_train, lam, tau, max_iter):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
#     n_iter  = 1e5
    n_loop  = 1
    _, _, FOM_time, beta, beta0 = loop_smoothing_hinge_loss('hinge', 
                                                      'l1', 
                                                      X_train, 
                                                      y_train, 
                                                      lam, 
                                                      tau, 
                                                      n_loop, 
                                                      max_iter, 
                                                      f, 
                                                      is_sparse=False)
    beta0 = beta0/np.sqrt(N)
    cons = np.ones(N) - y_train*( np.dot(X_train, beta) + beta0) 
    FOM_obj = np.sum([max(cons[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta))
    
    return FOM_obj, FOM_time, beta, beta0

# def use_Gurobi(X_train, y_train, lam):
#     N,P = np.shape(X_train)
#     beta = cp.Variable((P,1))
#     beta0 = cp.Variable()
#     y_train_ = np.reshape(y_train,(N,1))
#     loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train@beta + beta0)))
#     # print(X_train.shape)
#     print('Using single thread....\n\n')
#     # exit()
#     reg = cp.norm(beta, 1)
#     prob = cp.Problem(cp.Minimize(loss + lam*reg))
#     st = time.time()
#     env = gurobipy.Env()
#     env.setParam('Threads', 1)
#     gurobi_obj = prob.solve(solver = 'GUROBI', solver_opts = {"Threads": 1}, verbose=True)
#     ed = time.time()
#     gurobi_time = ed - st
    
#     return gurobi_obj, gurobi_time, np.reshape(beta.value, (P,)), beta0.value

def use_Gurobi(X_train, y_train, lam):
    init_time = time.time()
    n, p = np.shape(X_train)
    n_classes = len(np.unique(y_train))

    # Define the L1-MSVM linear model.
    MSVM_model = gp.Model("L1_binary_SVM")
    time_limit = 200000
    print('Using time limit: ', time_limit)
    MSVM_model.setParam('TimeLimit', time_limit)
    # MSVM_model.Params.LogToConsole = 0
    # MSVM_model.Params.OutputFlag = 0
    # MSVM_model.setParam('Threads', 1)

    # Define the variable.
    beta_plus = MSVM_model.addVars(p, lb=0, name="beta_plus")
    beta_minus = MSVM_model.addVars(p, lb=0, name="beta_minus")
    beta0 = MSVM_model.addVar(lb=-np.inf, name="beta0")
    eta = MSVM_model.addVars(n, lb=0, name="eta")

    # Define all the constraints.
    for j in range(0, n):
        MSVM_model.addConstr(eta[j] + y_train[j] * (gp.quicksum((X_train[j, i] * (beta_plus[i] - beta_minus[i])) for i in range(0, p)) + beta0) >= 1)

    # Define the objective.
    MSVM_model.setObjective(gp.quicksum(eta[i] for i in range(0, n)) + lam * gp.quicksum((beta_plus[i] + beta_minus[i]) for i in range(0, p)), GRB.MINIMIZE)

    # Solve the model
    MSVM_model.optimize()
    gurobi_time = time.time() - init_time

    beta_plus_val = np.zeros(p)
    beta_minus_val = np.zeros(p)
    for j in range(p):
        beta_plus_val[j] = beta_plus[j].x
        beta_minus_val[j] = beta_minus[j].x

    beta = beta_plus_val - beta_minus_val
    beta0_ = beta0.x
    objVal = MSVM_model.objVal

    return objVal, gurobi_time, beta, beta0_

