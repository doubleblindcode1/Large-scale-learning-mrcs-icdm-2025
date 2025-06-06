import csv
from os.path import dirname, join

import zipfile
import scipy.io
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

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

def load_letterrecog():
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'letterrecog.csv')
	X_full = np.loadtxt(data_file_name, delimiter=',')
	data = X_full[:,:-1]
	target = X_full[:,-1]
	return data, target

def load_cifar100():
	module_path = dirname(__file__)

	df = pd.read_csv(join(module_path,
							  'cifar100.csv'), header=None)
	dataset = np.array(df)
	data = dataset[:, :-1]
	target = dataset[:, -1]

	return data, target

def load_cats_vs_dogs(with_info=False):
	"""Load and return the Cats vs Dogs Data Set features extracted using a
	pretrained ResNet18 neural network (classification).

	===========================================
	Classes                                   2
	Samples per class             [11658,11604]
	Samples total                         23262
	Dimensionality                          512
	Features                              float
	===========================================

	Parameters
	----------
	with_info : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	bunch : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of Cats vs Dogs ResNet18 features
		csv dataset.

	(data, target) : tuple if ``with_info`` is False

	"""
	module_path = dirname(__file__)

	zf = zipfile.ZipFile(join(module_path,
							  'catsvsdogs_features_resnet18_1.csv.zip'))
	df1 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_1.csv'))
	zf = zipfile.ZipFile(join(module_path,
							  'catsvsdogs_features_resnet18_2.csv.zip'))
	df2 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_2.csv'))

	dataset = np.array(pd.concat([df1, df2]))
	data = dataset[:, :-1]
	target = dataset[:, -1]

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if not with_info:
		return data, normalizeLabels(target)

	return Bunch(data=data, target=normalizeLabels(target))

def load_yearbook_path():
	"""
	Returns the path of Yearbook Image Dataset
	"""
	module_path = dirname(__file__)
	path = join(module_path, 'data', 'yearbook')
	return path

def load_yearbook(with_info=False, with_attributes=False):
	"""Load and return the Yearbook Data Set features extracted using a
	pretrained ResNet18 neural network (classification).

	===========================================
	Classes                                   2
	Samples per class             [20248,17673]
	Samples total                         37921
	Dimensionality                          512
	Features                              float
	===========================================

	Parameters
	----------
	with_info : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	with_attributes : boolean, default=False.
		If True, returns an additional dictionary containing information of
		additional attributes: year, state, city, school of the portraits.
		The key 'attr_labels' in the dictionary contains these labels
		corresponding to each columns, while 'attr_data' corresponds to
		the attribute data in form of numpy array.

	Returns
	-------
	bunch : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of Yearbook ResNet18 features
		csv dataset.

	(data, target) : tuple if ``with_info`` is False

	"""
	module_path = dirname(__file__)

	zf = zipfile.ZipFile(join(module_path,
							  'yearbook_features_resnet18_1.csv.zip'))
	df1 = pd.read_csv(zf.open('yearbook_features_resnet18_1.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'yearbook_features_resnet18_2.csv.zip'))
	df2 = pd.read_csv(zf.open('yearbook_features_resnet18_2.csv'), header=None)

	dataset = np.array(pd.concat([df1, df2]))
	data = dataset[:, :-1]
	target = dataset[:, -1]

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	if with_attributes:
		attr = pd.read_csv(join(module_path, 'data',
								'yearbook_attributes.csv'))
		attr_labels = attr.columns.values
		attr_val = attr.values
		attr = {'attr_labels': attr_labels, 'attr_data': attr_val}

		if not with_info:
			return data, normalizeLabels(target), attr

		return Bunch(data=data, target=normalizeLabels(target),
					 attributes=attr, DESCR=descr_text)

	else:
		if not with_info:
			return data, normalizeLabels(target)

		return Bunch(data=data, target=normalizeLabels(target))

def load_real_sim():
	module_path = dirname(__file__)
	data_file_name = join(module_path, 'real_sim.binary')

	sep = "\t"

	dict_nnz = {}
	d = 20958
	def _process_data(row_array, dict_nnz, index):
		y_i = row_array[0]
		x_i = np.zeros(d)
		for j, val in enumerate(row_array[1:]):
			x_i[int(val.split(':')[0]) - 1] = float(val.split(':')[1])
			dict_nnz[index].append(int(val.split(':')[0]) - 1)
		return x_i, y_i, dict_nnz

	sp_data = []
	y = []
	index = 0
	with open(data_file_name) as csv_file:
		for row in csv_file:
			dict_nnz[index] = []
			data = row.split()
			x_i, y_i, dict_nnz = _process_data(data, dict_nnz, index)
			data = sps.csr_matrix(x_i)
			sp_data.append(data)
			y.append(y_i)
			index = index + 1


	sp_data = sps.vstack(sp_data)

	y = np.asarray(y)
	return sp_data, normalizeLabels(y), dict_nnz

def load_rcv1():
	module_path = dirname(__file__)
	data_file_name = join(module_path, 'rcv1.binary')

	sep = "\t"

	dict_nnz = {}
	d = 47236
	def _process_data(row_array, dict_nnz, index):
		y_i = row_array[0]
		x_i = np.zeros(d)
		for j, val in enumerate(row_array[1:]):
			x_i[int(val.split(':')[0]) - 1] = float(val.split(':')[1])
			dict_nnz[index].append(int(val.split(':')[0]) - 1)
		return x_i, y_i, dict_nnz

	sp_data = []
	y = []
	index = 0
	with open(data_file_name) as csv_file:
		for row in csv_file:
			dict_nnz[index] = []
			data = row.split()
			x_i, y_i, dict_nnz = _process_data(data, dict_nnz, index)
			data = sps.csr_matrix(x_i)
			sp_data.append(data)
			y.append(y_i)
			index = index + 1


	sp_data = sps.vstack(sp_data)

	y = np.asarray(y)
	return sp_data, normalizeLabels(y), dict_nnz

def load_news20():
	module_path = dirname(__file__)
	data_file_name = join(module_path, 'news20.binary')

	sep = "\t"

	dict_nnz = {}
	d = 1355191
	def _process_data(row_array, dict_nnz, index):
		y_i = row_array[0]
		x_i = np.zeros(d)
		for j, val in enumerate(row_array[1:]):
			x_i[int(val.split(':')[0]) - 1] = float(val.split(':')[1])
			dict_nnz[index].append(int(val.split(':')[0]) - 1)
		return x_i, y_i, dict_nnz

	sp_data = []
	y = []
	index = 0
	with open(data_file_name) as csv_file:
		for row in csv_file:
			dict_nnz[index] = []
			data = row.split()
			x_i, y_i, dict_nnz = _process_data(data, dict_nnz, index)
			data = sps.csr_matrix(x_i)
			sp_data.append(data)
			y.append(y_i)
			index = index + 1


	sp_data = sps.vstack(sp_data)

	y = np.asarray(y)
	return sp_data, normalizeLabels(y), dict_nnz

def load_page_blocks():
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'page_blocks.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = 5473
		n_features = 10
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		temp = next(data_file) # names of features
		feature_names = np.array(temp)

		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=float)
			target[i] = np.asarray(float(d[-1]), dtype=int)

	index = np.where(target == 7810937037581713463)[0][0]
	data = np.delete(data, index, 0)
	target = np.delete(target, index)
	index = np.where(target == 8214565725280534528)[0][0]
	data = np.delete(data, index, 0)
	target = np.delete(target, index)

	return data, normalizeLabels(target)

def load_pulsar():
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'pulsar.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = 17898
		n_features = 8
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		# temp = next(data_file) # names of features
		# feature_names = np.array(temp)

		# print(data_file.shape)
		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=float)
			target[i] = np.asarray(d[-1], dtype=int)

	return data, normalizeLabels(target)

def load_house16():
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'house16.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		# temp = next(data_file)
		n_samples = 22784
		n_features = 16
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		# temp = next(data_file) # names of features
		# feature_names = np.array(temp)

		# print(data_file.shape)
		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=float)
			target[i] = np.asarray(d[-1], dtype=int)

	return data, normalizeLabels(target)

def load_dry_bean():
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'dry_bean_dataset.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = 13611
		n_features = 16
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype="S20")
		# temp = next(data_file) # names of features
		# feature_names = np.array(temp)

		# print(data_file.shape)
		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=float)
			target[i] = d[-1]

	return data, normalizeLabels(target)

def load_mnist(with_info=False, split=False):
	"""Load and return the MNIST Data Set features extracted using a
	pretrained ResNet18 neural network (classification).

	======================= ===========================
	Classes                                          2
	Samples per class Train [5923,6742,5958,6131,5842,
							 5421,5918,6265,5851,5949]
	Samples per class Test    [980,1135,1032,1010,982,
								892,958,1028,974,1009]
	Samples total Train                          60000
	Samples total Test                           10000
	Samples total                                70000
	Dimensionality                                 512
	Features                                     float
	======================= ===========================

	Parameters
	----------
	with_info : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.
	split : boolean, default=False.
		If True, returns a dictionary instead of an array in the place of the
		data.

	Returns
	-------
	bunch : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of MNIST ResNet18 features
		csv dataset. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.

	(data, target) : tuple if ``with_info`` is False. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.
	"""
	module_path = dirname(__file__)

	zf = zipfile.ZipFile(join(module_path,
							  'mnist_features_resnet18_1.csv.zip'))
	df1 = pd.read_csv(zf.open('mnist_features_resnet18_1.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'mnist_features_resnet18_2.csv.zip'))
	df2 = pd.read_csv(zf.open('mnist_features_resnet18_2.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'mnist_features_resnet18_3.csv.zip'))
	df3 = pd.read_csv(zf.open('mnist_features_resnet18_3.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'mnist_features_resnet18_4.csv.zip'))
	df4 = pd.read_csv(zf.open('mnist_features_resnet18_4.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'mnist_features_resnet18_5.csv.zip'))
	df5 = pd.read_csv(zf.open('mnist_features_resnet18_5.csv'), header=None)

	dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
	data = dataset[:, :-1]
	target = dataset[:, -1]

	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	target = normalizeLabels(target)
	if not with_info:
		if split:
			# X_train, X_test, Y_train, Y_test
			X_train = data[:60000, :]
			Y_train = target[:60000]
			X_test = data[60000:, :]
			Y_test = target[60000:]
			return X_train, X_test, Y_train, Y_test
		else:
			return data, target
	else:
		if split:
			data = {'train': data[:60000, :], 'test': data[60000:, :]}
			target = {'train': target[:60000], 'test': target[60000:]}
		return Bunch(data=data, target=target, DESCR=descr_text)
	return 0

def load_fashion_mnist(with_info=False, split=False):
	"""Load and return the MNIST Data Set features extracted using a
	pretrained ResNet18 neural network (classification).

	======================= ===========================
	Classes                                          2
	Samples per class Train [5923,6742,5958,6131,5842,
							 5421,5918,6265,5851,5949]
	Samples per class Test    [980,1135,1032,1010,982,
								892,958,1028,974,1009]
	Samples total Train                          60000
	Samples total Test                           10000
	Samples total                                70000
	Dimensionality                                 512
	Features                                     float
	======================= ===========================

	Parameters
	----------
	with_info : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.
	split : boolean, default=False.
		If True, returns a dictionary instead of an array in the place of the
		data.

	Returns
	-------
	bunch : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of MNIST ResNet18 features
		csv dataset. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.

	(data, target) : tuple if ``with_info`` is False. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.
	"""
	module_path = dirname(__file__)

	zf = zipfile.ZipFile(join(module_path,
							  'fashion_mnist_features_resnet18_1.csv.zip'))
	df1 = pd.read_csv(zf.open('fashion_mnist_features_resnet18_1.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'fashion_mnist_features_resnet18_2.csv.zip'))
	df2 = pd.read_csv(zf.open('fashion_mnist_features_resnet18_2.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'fashion_mnist_features_resnet18_3.csv.zip'))
	df3 = pd.read_csv(zf.open('fashion_mnist_features_resnet18_3.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'fashion_mnist_features_resnet18_4.csv.zip'))
	df4 = pd.read_csv(zf.open('fashion_mnist_features_resnet18_4.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'fashion_mnist_features_resnet18_5.csv.zip'))
	df5 = pd.read_csv(zf.open('fashion_mnist_features_resnet18_5.csv'), header=None)

	dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
	data = dataset[:, :-1]
	target = dataset[:, -1]

	index = np.where(target == 512)[0][0]
	data = np.delete(data, index, 0)
	target = np.delete(target, index)
	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)

	target = normalizeLabels(target)
	if not with_info:
		if split:
			# X_train, X_test, Y_train, Y_test
			X_train = data[:60000, :]
			Y_train = target[:60000]
			X_test = data[60000:, :]
			Y_test = target[60000:]
			return X_train, X_test, Y_train, Y_test
		else:
			return data, target
	else:
		if split:
			data = {'train': data[:60000, :], 'test': data[60000:, :]}
			target = {'train': target[:60000], 'test': target[60000:]}
		return Bunch(data=data, target=target, DESCR=descr_text)
	return 0

def load_cifar10(with_info=False, split=False):
	"""Load and return the MNIST Data Set features extracted using a
	pretrained ResNet18 neural network (classification).

	======================= ===========================
	Classes                                          2
	Samples per class Train [5923,6742,5958,6131,5842,
							 5421,5918,6265,5851,5949]
	Samples per class Test    [980,1135,1032,1010,982,
								892,958,1028,974,1009]
	Samples total Train                          60000
	Samples total Test                           10000
	Samples total                                70000
	Dimensionality                                 512
	Features                                     float
	======================= ===========================

	Parameters
	----------
	with_info : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.
	split : boolean, default=False.
		If True, returns a dictionary instead of an array in the place of the
		data.

	Returns
	-------
	bunch : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of MNIST ResNet18 features
		csv dataset. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.

	(data, target) : tuple if ``with_info`` is False. If `split=False`, data is
		an array. If `split=True` data is a dictionary with 'train' and 'test'
		splits.
	"""
	module_path = dirname(__file__)

	zf = zipfile.ZipFile(join(module_path,
							  'cifar10_features_resnet18_1.csv.zip'))
	df1 = pd.read_csv(zf.open('cifar10_features_resnet18_1.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'cifar10_features_resnet18_2.csv.zip'))
	df2 = pd.read_csv(zf.open('cifar10_features_resnet18_2.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'cifar10_features_resnet18_3.csv.zip'))
	df3 = pd.read_csv(zf.open('cifar10_features_resnet18_3.csv'), header=None)
	zf = zipfile.ZipFile(join(module_path,
							  'cifar10_features_resnet18_4.csv.zip'))
	df4 = pd.read_csv(zf.open('cifar10_features_resnet18_4.csv'), header=None)

	dataset = np.array(pd.concat([df1, df2, df3, df4]))
	data = dataset[:, :-1]
	target = dataset[:, -1]

	data = np.delete(data, 50000, 0)
	target = np.delete(target, 50000)
	trans = SimpleImputer(strategy='median')
	data = trans.fit_transform(data)
	print(np.unique(target))
	# exit()

	target = normalizeLabels(target)
	if not with_info:
		if split:
			# X_train, X_test, Y_train, Y_test
			X_train = data[:60000, :]
			Y_train = target[:60000]
			X_test = data[60000:, :]
			Y_test = target[60000:]
			return X_train, X_test, Y_train, Y_test
		else:
			return data, target
	else:
		if split:
			data = {'train': data[:60000, :], 'test': data[60000:, :]}
			target = {'train': target[:60000], 'test': target[60000:]}
		return Bunch(data=data, target=target, DESCR=descr_text)
	return 0

def load_vehicle():
	"""Load and return the Iris Plants Dataset (classification).

	=================   =====================
	Classes                                 4
	Samples per class       [240,240,240,226]
	Samples total                         846
	Dimensionality                         18
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of the dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'vehicle.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		temp = next(data_file)  # names of features
		feature_names = np.array(temp[1:])

		classes = []
		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=np.float64)
			if d[-1] in classes:
				index = classes.index(d[-1])
				target[i] = np.asarray(index, dtype=int)
			else:
				classes.append(d[-1])
				target[i] = np.asarray(classes.index(d[-1]), dtype=int)

	return data, target

def load_satellite():
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                 6
	Samples per class               383, 307]
	Samples total                        6435
	Dimensionality                         36
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)


	data_file_name = join(module_path, 'satellite.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		temp = next(data_file)  # names of features
		feature_names = np.array(temp)

		for i, d in enumerate(data_file):
			try:
				data[i] = np.asarray(d[:-1], dtype=np.float64)
			except ValueError:
				print(i, d[:-1])
			target[i] = np.asarray(d[-1], dtype=int)

	return data, target

def load_optdigits():
	"""Load and return the Credit Approval prediction dataset (classification).

	=================   =====================
	Classes                                10
	Samples per class               383, 307]
	Samples total                        5620
	Dimensionality                         64
	Features             int, float, positive
	=================   =====================

	Parameters
	----------
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'target', the classification targets,
		'DESCR', the full description of the dataset,
		and 'filename', the physical location of adult csv dataset.

	(data, target) : tuple if ``return_X_y`` is True

	"""
	module_path = dirname(__file__)

	data_file_name = join(module_path, 'optdigits.csv')
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=int)
		temp = next(data_file)  # names of features
		feature_names = np.array(temp)

		for i, d in enumerate(data_file):
			try:
				data[i] = np.asarray(d[:-1], dtype=np.float64)
			except ValueError:
				print(i, d[:-1])
			target[i] = np.asarray(d[-1], dtype=int)

	return data, target

