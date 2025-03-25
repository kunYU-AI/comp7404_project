## 更新记录：
>  时间 | 更新文件名 | 更新内容 | 备注
1. 2025.03.26 | XGBoost_sparse.ipynb, model/XGBoost_process.py | 成功运行多种方法，XGBoost_process.py是模型代码，XGBoost_sparse.ipynb是实验加模型代码 | approximate（非sparse）的方法可以体现并行减少时间，其余方法多线程并行，额外增加算法耗时。原XGBoost标准库以C语言实现多线程，故python性能表现较差。
2. 2025.03.23 | XGBoost_sparse.ipynb model/XGBoost_scratch_multiprocess.py | 各类方法、多线程基本实现 | 可运行，但有问题。多线程和稀疏算法有问题。
3. 2025.03.20 | XGBoost.py | 加入_sparse_split_find()方法 | 用于在Allstate数据集上验证Sparsity-aware Split Finding算法。
