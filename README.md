README

The following folders implement the different versions of the proposed algorithm:

1) main_ccg_large_n - Implementation of the proposed algorithm corresponding to the scenario with large number of samples 
2) main_ccg_large_n_m - Implementation of the proposed algorithm corresponding to the scenario with large number of samples and features
3) main_ccg_large_n_multiclass - Memory efficient implementation of the proposed algorithm for multiclass setting with large number of samples.
4) main_ccg_large_n_m_sparse_binary - Memory efficient implementation of the proposed algorithm for sparse datasets with large number of samples and features corresponding to binary classification.

The datasets `pulsar` `house16` `rcv1` `satellite` `dry_bean` and `optdigits` are available as zip in the `Datasets` folder with functions to easily load them in `load.py`. The following python scripts can be used to reproduce the results of different methods on any dataset - 

1) run_ccg.py : Runs the proposed method MRC-CCG and the method SVM-CCG for any dataset.
2) run_lp.py : Runs the MRC-LP and SVM-LP method for any dataset
3) run_mrc_sub.py : Runs the MRC-SUB method for any dataset

Please install the libraries in requirements.txt and obtain a free academic license file for gurobi from https://www.gurobi.com/features/academic-named-user-license/ in order to run the code. To reproduce any experimental results corresponding to a dataset, run the following command - 

```
python3 <scriptname> <dataset> <random_seed>
```

The results are printed on the output console. The experimental results are averaged over random seeds 0, 1,..., 9. Note that large datasets are not uploaded to this repository but are easily available online. 

See https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ for real_sim and news20 dataset
See https://github.com/MachineLearningBCAM/Datasets/tree/main/data/multi_class_datasets for mnist, fashion_mnist, cifar10 datasets.
