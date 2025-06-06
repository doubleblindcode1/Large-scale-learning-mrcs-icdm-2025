import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

import itertools as it
import scipy.special as scs
from MRCpy.datasets import *
from Datasets.load import *

def normalizeLabels(origY):
	"""
	Normalize the labels of the instances in the range 0,...r-1 for r classes
	"""

	# Map the values of Y from 0 to r-1
	domY = np.unique(origY)
	Y = np.zeros(origY.shape[0], dtype=int)

	for i, y in enumerate(domY):
		Y[origY == y] = i

	return Y

def iteration_callback(model, where):
    if where == GRB.Callback.SIMPLEX:
        # Get the current iteration and time
        obj_val = model.cbGet(GRB.Callback.SPX_OBJVAL)

        # Save the times along the gurobi iterations
        model._solver_times.append(time.time() - model._start_time)

        # Save the upper bounds along the gurobi iterations
        model._R_k_gurobi.append(obj_val)

def mrc_lp_model_gurobi(F, b, tau_, lambda_, index_columns=None, nu_init=None, warm_start=None):
	"""
	Function to build and return the linear model of MRC 0-1 loss using the given
	constraint matrix and objective vector.

	Parameters:
	-----------
	F : array-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : array-like of shape (no_of_constraints)
		Right handside of the constraints.

	index_colums: array-like
		Selects the columns of the constraint matrix and objective vector.

	Return:
	-------
	model : A model object of MOSEK
		A solved MOSEK model of the MRC 0-1 linear model using the given constraints
		and objective.

	"""

	if index_columns is None:
		index_columns = np.arange(F.shape[1])

	time_limit = 200000
	print('\n\nUsing time limit: ', time_limit)
	print('\n\n')

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_primal")
	# MRC_model.Params.LogToConsole = 0
	# MRC_model.Params.OutputFlag = 0
	# MRC_model.setParam('Method', 0)
	MRC_model.setParam('TimeLimit', time_limit)
	MRC_model._start_time = time.time()
	MRC_model._solver_times = []
	MRC_model._R_k_gurobi = []

	# Define the variable.
	mu_plus = []
	mu_minus = []

	for i, index in enumerate(index_columns):
		mu_plus_i = MRC_model.addVar(lb=0, name="mu_+_" + str(index))
		mu_minus_i = MRC_model.addVar(lb=0, name="mu_-_" + str(index))

		if warm_start is not None:
			if warm_start[i] < 0:
				mu_minus_i.PStart = (-1) * warm_start[i]
				mu_plus_i.PStart = 0
			else:
				mu_plus_i.PStart = warm_start[i]
				mu_minus_i.PStart = 0

		mu_plus.append(mu_plus_i)
		mu_minus.append(mu_minus_i)

	nu_pos = MRC_model.addVar(name="nu_+")
	nu_neg = MRC_model.addVar(name="nu_-")

	if nu_init is not None:
		if nu_init < 0:
			nu_neg.PStart = (-1) * nu_init
			nu_pos.PStart = 0

		else:
			nu_pos.PStart = nu_init
			nu_neg.PStart = 0

	MRC_model.update()

	mu_plus = np.asarray(mu_plus)
	mu_minus = np.asarray(mu_minus)

	# Define all the constraints.
	for i in range(F.shape[0]):
		MRC_model.addConstr(F[i, index_columns] @ (mu_minus - mu_plus) +
							nu_pos - nu_neg >= b[i], "constr_" + str(i))

	# Define the objective.
	MRC_model.setObjective(tau_[index_columns] @ (mu_minus - mu_plus) +
						   lambda_[index_columns] @ (mu_minus + mu_plus) +
						   nu_pos - nu_neg, GRB.MINIMIZE)

	# Solve the model
	MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize(iteration_callback)

	return MRC_model

def load_dataset_(fig_name):
	# Loading the dataset
	dict_nnz = {}

	load_dataset = 'load_' + fig_name
	if fig_name == "news20" or fig_name == "rcv1" or fig_name == "real_sim":
		X, y, dict_nnz = eval(load_dataset + "()")
	else:
		X, y = eval(load_dataset + "()")

	print('The class distribution is: ')
	for y_i in np.unique(y):
		print('Class ' + str(y_i) + ': ' + str(np.sum(y == y_i)))

	y = normalizeLabels(y)
	n, d = X.shape
	n_classes = len(np.unique(y))
	print('Dataset + ' + str(fig_name) + \
		  ' loaded. The dimensions are : ' \
		  + str(n) + ', ' + str(d))
	print('Number of classes: ', n_classes)

	return X, y, n, d, n_classes, dict_nnz

def predict_proba(X_transform, mu_, nu_, n_classes, large_n):
	'''
	Conditional probabilities corresponding to each class
	for each unlabeled input instance

	Parameters
	----------
	X : `array`-like of shape (`n_samples`, `n_dimensions`)
		Testing instances for which
		the prediction probabilities are calculated for each class.

	Returns
	-------
	hy_x : `ndarray` of shape (`n_samples`, `n_classes`)
		Probabilities :math:`(p(y|x))` corresponding to the predictions
		for each class.

	'''

	# Unnormalized conditional probabilityes
	if large_n == 1 and n_classes > 2:
		d = int(mu_.shape[0] / n_classes)
		mu_ = np.reshape(mu_, (n_classes, d))
		hy_x = np.clip(1 + np.dot(X_transform, mu_.T) + nu_, 0., None)
	else:
		hy_x = np.clip(1 + np.dot(X_transform, mu_) + nu_, 0., None)

	# normalization constraint
	c = np.sum(hy_x, axis=1)
	# check when the sum is zero
	zeros = np.isclose(c, 0)
	c[zeros] = 1
	hy_x[zeros, :] = 1 / n_classes
	c = np.tile(c, (n_classes, 1)).transpose()
	hy_x = hy_x / c

	return hy_x

def predict_proba_sparse(X_transform, mu_, nu_, n_classes):
	'''
	Conditional probabilities corresponding to each class
	for each unlabeled input instance

	Parameters
	----------
	X : `array`-like of shape (`n_samples`, `n_dimensions`)
		Testing instances for which
		the prediction probabilities are calculated for each class.

	Returns
	-------
	hy_x : `ndarray` of shape (`n_samples`, `n_classes`)
		Probabilities :math:`(p(y|x))` corresponding to the predictions
		for each class.

	'''

	# Unnormalized conditional probabilityes
	hy_x1 = np.clip(1 + X_transform.dot(mu_) + nu_, 0., None)
	hy_x2 = np.clip(1 - X_transform.dot(mu_) + nu_, 0., None)
	hy_x = np.asarray([hy_x1, hy_x2]).T

	# normalization constraint
	c = np.sum(hy_x, axis=1)
	# check when the sum is zero
	zeros = np.isclose(c, 0)
	c[zeros] = 1
	hy_x[zeros, :] = 1 / n_classes
	c = np.tile(c, (n_classes, 1)).transpose()
	hy_x = hy_x / c

	return hy_x

# Multi-class L1-SVM gurobi
def l1_msvm_gurobi_lp(X_train, y_train, lam):

	init_time = time.time()
	n, p = np.shape(X_train)
	n_classes = len(np.unique(y_train))

	# Define the L1-MSVM linear model.
	MSVM_model = gp.Model("L1_Multiclass_SVM")
	MSVM_model.Params.LogToConsole = 0
	MSVM_model.Params.OutputFlag = 0

	# Define the variable.
	beta_plus = MSVM_model.addVars(n_classes, p, lb=0, name="beta_plus")
	beta_minus = MSVM_model.addVars(n_classes, p, lb=0, name="beta_minus")
	beta0_plus = MSVM_model.addVars(n_classes, lb=0, name="beta0_plus")
	beta0_minus = MSVM_model.addVars(n_classes, lb=0, name="beta0_minus")
	eta = MSVM_model.addVars(n * (n_classes - 1), lb=0, name="eta")

	# Define all the constraints.
	for i in range(0, n):
		k = 0
		for j in range(0, n_classes):
			if j != y_train[i]:
				MSVM_model.addConstr(gp.quicksum(((beta_plus[j, m] - beta_minus[j, m]) * X_train[i, m]) for m in range(p)) + (beta0_plus[j] - beta0_minus[j]) + 1 <= eta[(i * (n_classes - 1)) + k])
				k = k + 1

	# Regularization constraint
	MSVM_model.addConstr(gp.quicksum((beta_plus[i, j] + beta_minus[i, j]) for i in range(0, n_classes) for j in range(p)) <= lam)

	# Class coefficient equal to zero constraint.
	for j in range(0, p):
		MSVM_model.addConstr(gp.quicksum((beta_plus[i, j] - beta_minus[i, j]) for i in range(0, n_classes)) == 0)


	MSVM_model.addConstr(gp.quicksum((beta0_plus[i] - beta0_minus[i]) for i in range(0, n_classes)) == 0)

	# Define the objective.
	MSVM_model.setObjective(gp.quicksum(eta[i] for i in range(0, n * (n_classes - 1))), GRB.MINIMIZE)

	# Solve the model
	MSVM_model.optimize()

	beta_plus_val = np.zeros((n_classes, p))
	beta_minus_val = np.zeros((n_classes, p))
	beta0_plus_val = np.zeros(n_classes)
	beta0_minus_val = np.zeros(n_classes)
	for i in range(n_classes):
		for j in range(p):
			beta_plus_val[i, j] = beta_plus[i, j].x
			beta_minus_val[i, j] = beta_minus[i, j].x

	for i in range(n_classes):
		beta0_plus_val[i] = beta0_plus[i].x
		beta0_minus_val[i] = beta0_minus[i].x

	beta = beta_plus_val - beta_minus_val
	beta0 = beta0_plus_val - beta0_minus_val
	objVal = MSVM_model.objVal

	return beta, beta0, objVal

# MRCs gurobi
def gurobi_lpsolve_mrc_primal(X_train, X_test, y_test, phi_ob, tau_, lambda_):

	print('Solving the MRC problem using gurobi LP')
	init_time = time.time()
	n = X_train.shape[0]
	phi_ = phi_ob.eval_x(X_train)

	F_ = np.vstack(list(np.sum(phi_[:, S, ], axis=1)
					for numVals in range(1, phi_ob.n_classes + 1)
					for S in it.combinations(np.arange(phi_ob.n_classes), numVals)))

	cardS = np.arange(1, phi_ob.n_classes + 1). \
		repeat([n * scs.comb(phi_ob.n_classes, numVals)
				for numVals in np.arange(1, phi_ob.n_classes + 1)])

	# Constraint coefficient matrix
	F = F_ / (cardS[:, np.newaxis])
	
	# The bounds on the constraints
	b = 1 - (1 / cardS)
	I = np.arange(F.shape[1])

	# Build the gurobi model and solve it.
	MRC_model = mrc_lp_model_gurobi(F,
									b,
									tau_,
									lambda_,
									I,
									None,
									None)

	# Get the upper bound
	upper_bound = MRC_model.objVal
	print('The worst-case error is: ', upper_bound)

	end_time = time.time() - init_time

	solver_times = []
	R_k_gurobi = []

	# with threading.Lock():
	for times in MRC_model._solver_times:
		solver_times.append(times)
	for R_k in MRC_model._R_k_gurobi:
		R_k_gurobi.append(R_k)

	# Get the solution and the classification error
	if MRC_model.Status == 2:
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))).x for i in I])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))).x for i in I])
		nu_pos = MRC_model.getVarByName("nu_+").x
		nu_neg = MRC_model.getVarByName("nu_-").x
		mu = mu_plus - mu_minus
		nu = nu_pos - nu_neg
	
		# Compute classification error
		if phi_ob.n_classes == 2:
			hy_x = predict_proba(phi_ob.eval_x(X_test), mu, nu, phi_ob.n_classes, 0)
		else:
			X_test_transform = np.hstack(([[1]] * X_test.shape[0], phi_ob.transform(X_test)))
			hy_x = predict_proba(X_test_transform, mu, nu, phi_ob.n_classes, 1)
		clf_error = np.average(y_test != np.argmax(hy_x, axis=1))
		print('The classification error is: ', clf_error)
	else:
		clf_error = 0

	return upper_bound, clf_error, end_time, solver_times, R_k_gurobi
