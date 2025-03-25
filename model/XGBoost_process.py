# 默认处理空值为-1和np.nan
# 修改空值处理：搜索missing_mask和miss_mask

import pandas as pd
import numpy as np
from numba import njit, set_num_threads, prange

import math
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.sparse import csr_matrix, csc_matrix
import matplotlib.pyplot as plt
from collections import defaultdict

import os

@njit(nogil=True)
def compute_rank_numba(values, weights, epsilon):
    total_weight = np.sum(weights)
    target = epsilon * total_weight
    current_weight = 0.0
    candidates = []
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    for i in prange(len(sorted_values)):
        current_weight += sorted_weights[i]
        if current_weight >= target:
            candidates.append(sorted_values[i])
            current_weight = 0.0
    return np.array(candidates)
    
#---------------------------------#
# exact split
@njit(nogil=True)
def compute_split_gain(g_left, h_left, g_total, h_total, reg_lambda, gamma, min_child_weight):
    """计算单个分割点的增益"""
    if (h_left < min_child_weight) or ((h_total - h_left) < min_child_weight):
        return 0.0
    left_gain = g_left * g_left / (h_left + reg_lambda)
    right_gain = (g_total - g_left) * (g_total - g_left) / (h_total - h_left + reg_lambda)
    root_gain = g_total * g_total / (h_total + reg_lambda)
    return 0.5 * (left_gain + right_gain - root_gain) - gamma

@njit(nogil=True)
def find_split_exact_single_feature(values, g_sorted, h_sorted, 
                                  total_g, total_h, reg_lambda, gamma, min_child_weight):
    """计算单个特征的最佳分割点"""
    g_cum = np.cumsum(g_sorted)
    h_cum = np.cumsum(h_sorted)
    
    best_gain = 0.0
    best_threshold = 0.0
    
    for j in range(1, len(values)):
        if values[j] == values[j-1]:
            continue
            
        gain = compute_split_gain(g_cum[j-1], h_cum[j-1], 
                                total_g, total_h,
                                reg_lambda, gamma, min_child_weight)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = (values[j-1] + values[j]) / 2
    
    return best_gain, best_threshold
#-------------------------------------#
# approximate split
@njit(nogil=True)
def find_split_approx_single_feature(values, local_indices, g, h, candidates,
                                   total_g, total_h, reg_lambda, gamma, min_child_weight):
    """计算单个特征的最佳近似分割点"""
    best_gain = 0.0
    best_threshold = None
    
    for thresh in candidates:
        left_mask = values <= thresh
        sum_g_left = g[local_indices[left_mask]].sum()
        sum_h_left = h[local_indices[left_mask]].sum()
        sum_h_right = total_h - sum_h_left
        
        if sum_h_left < min_child_weight or sum_h_right < min_child_weight:
            continue
            
        gain = compute_split_gain(sum_g_left, sum_h_left,
                                total_g, total_h,
                                reg_lambda, gamma, min_child_weight)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = thresh
    
    return best_gain, best_threshold
#-------------------------------------#
# sparse split
@njit(nogil=True)
def _process_sparse_feature(values, g_non_miss, h_non_miss, sum_g_miss, sum_h_miss, 
                          total_g, total_h, reg_lambda, gamma, min_child_weight):
    """使用 numba 加速稀疏特征的预处理计算"""
    g_cum = np.cumsum(g_non_miss)
    h_cum = np.cumsum(h_non_miss)
    return g_cum, h_cum
    
@njit(nogil=True)
def find_split_sparse_exact_single_feature_fast(values, g_cum, h_cum,
                                              sum_g_miss, sum_h_miss,
                                              total_g, total_h, reg_lambda, gamma, min_child_weight):
    """优化后的稀疏特征精确分割算法"""
    best_gain = 0.0
    best_threshold = None
    best_default_left = None
    has_missing = (sum_g_miss != 0) or (sum_h_miss != 0)
    
    # 预分配数组以避免重复计算
    gains_left = np.zeros(len(values), dtype=np.float32)
    gains_right = np.zeros(len(values), dtype=np.float32)
    
    # 并行计算所有可能的分割点的增益
    for i in prange(1, len(values)):
        if values[i] == values[i-1]:
            continue
            
        sum_g_left = g_cum[i-1]
        sum_h_left = h_cum[i-1]
        sum_h_right = total_h - sum_h_left - sum_h_miss
        
        if sum_h_left < min_child_weight or sum_h_right < min_child_weight:
            continue
        
        if has_missing:
            gains_left[i] = compute_split_gain(
                sum_g_left + sum_g_miss, sum_h_left + sum_h_miss,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            gains_right[i] = compute_split_gain(
                sum_g_left, sum_h_left,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            current_gain = max(gains_left[i], gains_right[i])
            current_default_left = gains_left[i] > gains_right[i]
        else:
            current_gain = compute_split_gain(
                sum_g_left, sum_h_left,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            current_default_left = True
        
        if current_gain > best_gain:
            best_gain = current_gain
            best_threshold = (values[i-1] + values[i]) / 2
            best_default_left = current_default_left if has_missing else True
            
    return best_gain, best_threshold, best_default_left

@njit(nogil=True)
def find_split_sparse_approx_single_feature_fast(values, g_cum, h_cum,
                                               sum_g_miss, sum_h_miss, candidates,
                                               total_g, total_h, reg_lambda, gamma, min_child_weight):
    """优化后的稀疏特征近似分割算法"""
    best_gain = 0.0
    best_threshold = None
    best_default_left = None
    has_missing = (sum_g_miss != 0) or (sum_h_miss != 0)
    
    # 预分配数组以避免重复计算
    n_candidates = len(candidates)
    gains_left = np.zeros(n_candidates, dtype=np.float32)
    gains_right = np.zeros(n_candidates, dtype=np.float32)
    
    # 预计算每个候选点的左侧累积和
    for i in prange(n_candidates):
        thresh = candidates[i]
        left_idx = np.searchsorted(values, thresh, side='right') - 1
        
        if left_idx < 0:
            continue
            
        sum_g_left = g_cum[left_idx]
        sum_h_left = h_cum[left_idx]
        sum_h_right = total_h - sum_h_left - sum_h_miss
        
        if sum_h_left < min_child_weight or sum_h_right < min_child_weight:
            continue
        
        if has_missing:
            # 计算将缺失值分到左边的增益
            gains_left[i] = compute_split_gain(
                sum_g_left + sum_g_miss, sum_h_left + sum_h_miss,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            # 计算将缺失值分到右边的增益
            gains_right[i] = compute_split_gain(
                sum_g_left, sum_h_left,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            current_gain = max(gains_left[i], gains_right[i])
            current_default_left = gains_left[i] > gains_right[i]
        else:
            current_gain = compute_split_gain(
                sum_g_left, sum_h_left,
                total_g, total_h, reg_lambda, gamma, min_child_weight
            )
            current_default_left = True
        
        if current_gain > best_gain:
            best_gain = current_gain
            best_threshold = thresh
            best_default_left = current_default_left if has_missing else True
            
    return best_gain, best_threshold, best_default_left

#-----------------------------------#
# 分位图草图
class WeightedQuantileSketch:
    def __init__(self, eps=0.05, max_size=100):
        self.eps = eps
        self.max_size = max_size
        self.summaries = defaultdict(list)
    
    def _compute_rank(self, values, weights, epsilon):
        return compute_rank_numba(values, weights, epsilon)
    
    def create_summary(self, feature_idx, values, hessians):
        candidates = self._compute_rank(values, hessians, self.eps)
        self.summaries[feature_idx] = candidates
    
    def get_candidates(self, feature_idx):
        return self.summaries.get(feature_idx, [])

# 压缩列存储块结构CSC
class ColumnBlock:
    def __init__(self, data, feature_idx, sample_indices):
        self.feature_idx = feature_idx
        self.sample_indices = sample_indices  # 当前节点的样本索引（指向原始数据）
        self.data = data[self.sample_indices, feature_idx].astype(np.float32)

        self.sorted_idx = np.argsort(self.data)  # 对当前子集排序的局部索引（0到len(sample_indices)-1）
        self.sorted_values = self.data[self.sorted_idx]

        # one-hot encode 用-1表示缺失值，避免0与连续变量的值混淆
        self.missing_mask = np.isnan(self.sorted_values) | (self.sorted_values == -1.0)

    def get_subset(self, sub_indices,exact=True):
        # 返回sorted_values和sorted sample_indices的子集
        sub_mask = np.isin(self.sample_indices[self.sorted_idx], sub_indices)
        if exact:
            return self.sorted_values[sub_mask], self.sample_indices[self.sorted_idx[sub_mask]]
        else:
            return self.sorted_values[sub_mask], self.sorted_idx[sub_mask]
            
    def get_missing(self, sub_indices):
        sub_mask = np.isin(self.sample_indices[self.sorted_idx], sub_indices)
        valid_mask = (~self.missing_mask) & sub_mask
        missing_mask = self.missing_mask & sub_mask
        return (self.sorted_values[valid_mask], 
                self.sample_indices[self.sorted_idx[valid_mask]],
                self.sample_indices[self.sorted_idx[missing_mask]])


class BoostTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, default_left=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.default_left = default_left
        self.best_gain = 0.0

class BoosterTree:
    def __init__(self, params, n_threads=1): 
        self.root = None

        self.max_depth = params.get('max_depth', 5)
        self.reg_lambda = params.get('reg_lambda', 1.0)
        self.gamma = params.get('gamma', 0.0)
        self.min_child_weight = params.get('min_child_weight', 1.0)
        self.colsample_bynode = params.get('colsample_bynode', 1.0)
        self.method = params.get('method', 'exact')
        self.sparse = params.get('sparse', False)

        self.column_blocks = None  # 每棵树使用column blocks来并行
        self.X = None 
        self.grad_stats = None  
        self.quantile_sketch = WeightedQuantileSketch(eps=0.05) if self.method == 'approx' else None

        self.n_threads = n_threads

    def fit(self, X, grad_stats, indices=None):
        if indices is None:
            indices = np.arange(len(grad_stats))
        
        self.X = X
        self.grad_stats = grad_stats
        self.n_samples, self.n_features = len(indices), X.shape[1]
        
        # 初始化列块存储
        self.column_blocks = [ColumnBlock(self.X, fidx, indices) for fidx in range(self.n_features)]

        # 近似分位算法
        if self.method == 'approx':
            self._prepare_quantile_sketches(indices)
    
        self.root = self._grow_tree(indices, depth=0)
    
    def _prepare_quantile_sketches(self, indices):
        """并行处理特征的分位数统计"""
        self.non_missing_masks = []
        X_subset = self.X[indices]
        h = self.grad_stats[indices, 1]

        def process_feature(fidx):
            try:
                feature_values = X_subset[:, fidx]
                non_missing_mask = ~np.isnan(feature_values)
                values = feature_values[non_missing_mask]
                h_non_missing = h[non_missing_mask]
                return fidx, non_missing_mask, values, h_non_missing
            except Exception as e:
                print(f"处理特征 {fidx} 时出错: {e}")
                return fidx, None, None, None

        # 使用线程池并行处理特征
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for fidx in range(self.n_features):
                futures.append(executor.submit(process_feature, fidx))
            
            # 收集结果并创建摘要
            self.non_missing_masks = [None] * self.n_features
            for future in futures:
                fidx, mask, values, h_values = future.result()
                if mask is not None:
                    self.non_missing_masks[fidx] = mask
                    if len(values) > 0:
                        self.quantile_sketch.create_summary(fidx, values, h_values)

    def _grow_tree(self, sample_indices, depth=0):
        g = self.grad_stats[sample_indices, 0]
        h = self.grad_stats[sample_indices, 1]
        node_value = -g.sum() / (h.sum() + self.reg_lambda)
        node = BoostTreeNode(value=node_value)
        
        if depth >= self.max_depth or len(sample_indices) < 2:
            return node
    
        # 随机特征子集
        feature_indices = self._get_feature_subset()
        
        if self.method == 'exact' and not self.sparse:
            best_gain, best_feature, best_threshold = self._find_split_exact(sample_indices, g, h, feature_indices)
            if best_gain <= 0.0:
                return node
            node.feature_idx, node.threshold, node.best_gain = best_feature, best_threshold, best_gain
            node.default_left = False
            feature_values = self.X[sample_indices, best_feature]
            left_mask = feature_values <= best_threshold
            right_mask = ~left_mask

        elif self.method == 'approx' and not self.sparse:
            best_gain, best_feature, best_threshold = self._find_split_approx(sample_indices, g, h, feature_indices)
            if best_gain <= 0.0:
                return node
            node.feature_idx, node.threshold, node.best_gain = best_feature, best_threshold, best_gain
            node.default_left = False
            feature_values = self.X[sample_indices, best_feature]
            left_mask = feature_values <= best_threshold
            right_mask = ~left_mask
        
        else:  # Sparse-aware (exact or approx)
            best_gain, best_feature, best_threshold, best_default_left = self._find_split_sparse(
                sample_indices, g, h, feature_indices, exact=(self.method == 'exact')
            )
            if best_gain <= 0.0:
                return node
            node.feature_idx, node.threshold, node.best_gain = best_feature, best_threshold, best_gain
            node.default_left = best_default_left
            feature_values = self.X[sample_indices, best_feature]
            miss_mask = np.isnan(feature_values) | (feature_values == -1.0)
            left_mask = ((feature_values <= best_threshold) & ~miss_mask) | (miss_mask & best_default_left)
            right_mask = ~left_mask
        
        node.left = self._grow_tree(sample_indices[left_mask], depth + 1)
        node.right = self._grow_tree(sample_indices[right_mask], depth + 1)
        return node

    def _get_feature_subset(self):
        n_selected = int(self.n_features * self.colsample_bynode)
        return np.random.choice(self.n_features, n_selected, replace=False)

    def _find_split_exact(self, sample_indices, g, h, feature_indices):
        """使用多线程并行计算最佳分割点"""
        total_g, total_h = g.sum(), h.sum()
        def process_feature(fidx):
            block = self.column_blocks[fidx]
            values, local_indices = block.get_subset(sample_indices, exact=True)
            g_sorted = self.grad_stats[local_indices,0]
            h_sorted = self.grad_stats[local_indices,1]

            gain, threshold = find_split_exact_single_feature(
                values,
                g_sorted, h_sorted,
                total_g, total_h,
                self.reg_lambda,
                self.gamma,
                self.min_child_weight
            )
            
            return gain, fidx, threshold
        
        # 多线程处理
        with ThreadPoolExecutor(self.n_threads) as executor:
            results = list(executor.map(process_feature, feature_indices))
        
        # 找出最佳分割点
        best_result = max(results, key=lambda x: x[0])
        return best_result[0], int(best_result[1]), best_result[2]    

    def _find_split_approx(self, sample_indices, g, h, feature_indices):
        """使用多线程并行计算最佳近似分割点"""
        total_g, total_h = g.sum(), h.sum()
        
        def process_feature(fidx):
            block = self.column_blocks[fidx]
            values, local_indices = block.get_subset(sample_indices, exact=False)
            if len(values) == 0:
                return 0.0, fidx, None

            candidates = self.quantile_sketch.get_candidates(fidx)
            if len(candidates) == 0:
                return 0.0, fidx, None
            
            gain, threshold = find_split_approx_single_feature(
                values, local_indices, g, h, candidates,
                total_g, total_h,
                self.reg_lambda, self.gamma, self.min_child_weight
            )
            
            return gain, fidx, threshold
        
        # 多线程处理
        with ThreadPoolExecutor(self.n_threads) as executor:
            results = list(executor.map(process_feature, feature_indices))
        
        # 找出最佳分割点
        best_result = max(results, key=lambda x: x[0])
        return best_result[0], int(best_result[1]), best_result[2]

    def _find_split_sparse(self, sample_indices, g, h, feature_indices, exact=True):
        """优化后的稀疏特征分割点查找"""
        total_g, total_h = g.sum(), h.sum()
    
        def process_feature(fidx):
            block = self.column_blocks[fidx]
            values, non_missing_indices, missing_indices = block.get_missing(sample_indices)
            if len(values) == 0:
                return 0.0, fidx, None, None
            
            # 预计算梯度统计量
            g_non_miss = self.grad_stats[non_missing_indices,0]
            h_non_miss = self.grad_stats[non_missing_indices,1]
            sum_g_miss = np.sum(self.grad_stats[missing_indices,0]) if len(missing_indices) > 0 else 0.0
            sum_h_miss = np.sum(self.grad_stats[missing_indices,1]) if len(missing_indices) > 0 else 0.0
            
            # 使用 numba 加速的预处理
            g_cum, h_cum = _process_sparse_feature(
                values, g_non_miss, h_non_miss,
                sum_g_miss, sum_h_miss,
                total_g, total_h,
                self.reg_lambda, self.gamma,
                self.min_child_weight
            )
            
            if exact:
                gain, threshold, default_left = find_split_sparse_exact_single_feature_fast(
                    values, g_cum, h_cum,
                    sum_g_miss, sum_h_miss,
                    total_g, total_h,
                    self.reg_lambda, self.gamma, self.min_child_weight
                )
            else:
                candidates = self.quantile_sketch.get_candidates(fidx)
                gain, threshold, default_left = find_split_sparse_approx_single_feature_fast(
                    values, g_cum, h_cum,
                    sum_g_miss, sum_h_miss, candidates,
                    total_g, total_h,
                    self.reg_lambda, self.gamma, self.min_child_weight
                )
            
            return gain, fidx, threshold, default_left

        # 多线程处理
        with ThreadPoolExecutor(self.n_threads) as executor:
            results = list(executor.map(process_feature, feature_indices))
        
        # 找出最佳分割点
        best_result = max(results, key=lambda x: x[0])
        return best_result[0], int(best_result[1]), best_result[2], best_result[3]  
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.left is None and node.right is None:
            return node.value
        if np.isnan(x[node.feature_idx]):
            return self._predict(x, node.left if node.default_left else node.right)
        return self._predict(x, node.left if x[node.feature_idx] <= node.threshold else node.right)

    def release_memory(self):
        # 显式释放内存
        del self.X
        del self.column_blocks
        del self.grad_stats 
        del self.quantile_sketch

# XGBoost模型
class XGBoostModel:
    def __init__(self, params=None, random_seed=None, n_threads=1):
        params = params or {}
        self.params = params
        self.base_value = params.get('base_value', 0.5)
        self.subsample = params.get('subsample', 1.0)
        self.learning_rate = params.get('learning_rate', 0.3)

        self.n_threads = min(n_threads, os.cpu_count())
        set_num_threads(self.n_threads)
        self.rng = np.random.default_rng(random_seed)
        self.boosters = []
        
        self.avg_time_per_tree = None
    
    def fit(self, X, y, objective, rounds, verbose=False):
        if isinstance(X, (csr_matrix, csc_matrix)):
            X = X.toarray()
        y = np.asarray(y)
        n_samples = len(y)

        preds = np.full(n_samples, self.base_value, dtype=np.float32)
        grad_stats = np.empty((n_samples, 2), dtype=np.float32)
        times = []

        for i in range(rounds):
            start_time = time.time()
            grad_stats[:, 0] = objective.gradient(y, preds)
            grad_stats[:, 1] = objective.hessian(y, preds)
            
            # 随机选择数据子集
            if self.subsample < 1.0:
                sample_size = math.floor(self.subsample * len(y))
                indices = self.rng.choice(len(y), size=sample_size, replace=False)
            else:
                indices = None
            
            # 训练单个树
            booster = BoosterTree(self.params, n_threads=self.n_threads)
            booster.fit(X, grad_stats, indices)
            
            preds += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
            end_time = time.time()
            times.append(end_time - start_time)
            
            # 释放内存
            booster.release_memory()
            
            if verbose:
                current_loss = objective.loss(y, preds)
                print(f"[{i}] Training Loss: {current_loss:.5f}")
        self.avg_time_per_tree = np.mean(times)

    def predict(self, X):
        if isinstance(X, (csr_matrix, csc_matrix)):
            X = X.toarray()
        # 初始化预测值
        predictions = np.full(len(X), self.base_value, dtype=np.float32)
        
        # 批量计算树的预测值
        for booster in self.boosters:
            predictions += self.learning_rate * booster.predict(X)
        return predictions