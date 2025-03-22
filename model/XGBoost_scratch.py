# 相比于决策树，XGBoost的区别：
# 1.Gradient Boost算法
# 2.正则化
# 3.split finding algorithm (scoring function): exact 和 approximate
# 4.out-of-core算法
import pandas as pd
import numpy as np
import math

class XGBoostModel:
    def __init__(self, params=None, random_seed=None):
        params = params or {}
        self.params = params
        # XGBoost Model hyperparameters
        self.base_value = params.get('base_value', 0.5)
        self.subsample = params.get('subsample', 1.0) # 样本子采样比例
        self.learning_rate = params.get('learning_rate', 0.3) # 单个树学习率
        # 子采样的随机数生成器
        self.rng = np.random.default_rng(random_seed)
        # 存储所有的树
        self.boosters = []

    def fit(self, X, y, objective, rounds, verbose=False):
        # 第一颗树的基础预测值
        preds = self.base_value * np.ones(shape=y.shape)
        
        for i in range(rounds):
            grad = objective.gradient(y, preds)
            hess = objective.hessian(y, preds)
            
            # 随机选择子数据集
            indices = self._get_subsample_indices(y) if self.subsample < 1.0 else None
            
            # 训练单个树
            booster = BoosterTree(self.params)
            booster.fit(X.values, grad, hess, indices)
            
            # 加总预测值，以学习率为权重
            preds += self.learning_rate * booster.predict(X.values)
            self.boosters.append(booster)
            
            # 打印每棵树的训练损失
            if verbose:
                current_loss = objective.loss(y, preds)
                print(f"[{i}] Training Loss: {current_loss:.5f}")

    def predict(self, X):
        # 确保传入的X为numpy数组
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        adjustments = self.learning_rate * np.sum([t.predict(X_array) for t in self.boosters], axis=0)
        return self.base_value + adjustments

    def _get_subsample_indices(self, y):
        sample_size = math.floor(self.subsample * len(y))
        return self.rng.choice(len(y), size=sample_size, replace=False)

class BoostTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # 分裂特征索引
        self.threshold = threshold      # 分裂阈值
        self.left = left                # 左子树
        self.right = right              # 右子树
        self.value = value              # 节点weight
        self.best_gain = 0.0            # 最佳分裂的loss reduction

class BoosterTree:
    def __init__(self, params):
        self.root = None # 根节点
        self.max_depth = params.get('max_depth', 5) # 最大深度
        self.reg_lambda = params.get('reg_lambda', 1.0) # 正则化参数
        self.gamma = params.get('gamma', 0.0) # 约束树的大小（叶节点的数量）
        self.min_child_weight = params.get('min_child_weight', 1.0) # 最小样本权重和
        self.colsample_bynode = params.get('colsample_bynode', 1.0) # 特征采样比例（修正参数名）
        self.method = params.get('method', 'exact') # 分裂寻找方法：exact, approximate, sparse
        
    def fit(self, X, g, h, indices=None):
        """拟合梯度提升树
        Args:
            X: 特征矩阵（numpy数组）
            g: 一阶梯度
            h: 二阶hessian
        """
        if isinstance(g, pd.Series): g = g.values
        if isinstance(h, pd.Series): h = h.values
        if indices is None: indices = np.arange(len(g))
        
        g, h = g[indices], h[indices]
        self.n_samples, self.n_features = len(indices), X.shape[1]
        self.root = self._grow_tree(X[indices], g, h, depth=0)
    
    def _grow_tree(self, X, g, h, depth):
        # 当前节点值
        node_value = -g.sum() / (h.sum() + self.reg_lambda)
        node = BoostTreeNode(value=node_value)
        
        # 终止条件: 达到最大深度或样本不足
        if depth >= self.max_depth or X.shape[0] < 2:
            return node
        
        # 特征采样
        feature_indices = self._get_feature_subset()
        
        # 寻找最佳分裂
        if self.method == 'exact':
            best_gain, best_feature, best_threshold = self._exact_split_find(X, g, h, feature_indices)
        elif self.method == 'approximate':
            best_gain, best_feature, best_threshold = self._approximate_split_find(X, g, h, feature_indices)
        elif self.method == 'sparse':
            best_gain, best_feature, best_threshold = self._sparse_split_find(X, g, h, feature_indices)
        
        # 如果增益不足则作为叶节点
        if best_gain == 0.0:
            return node
        
        # 执行分裂
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        # 递归构建子树
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.best_gain = best_gain
        node.left = self._grow_tree(X[left_idx], g[left_idx], h[left_idx], depth+1)
        node.right = self._grow_tree(X[right_idx], g[right_idx], h[right_idx], depth+1)
        
        return node
    
    def _get_feature_subset(self):
        """获得特征子集"""
        n_selected = int(self.n_features * self.colsample_bynode)
        return np.random.choice(self.n_features, n_selected, replace=False)
    
    def _exact_split_find(self, X, g, h, feature_indices):
        """寻找最佳分裂点"""
        best_gain = 0.0
        best_feature, best_threshold = None, None
        total_g = g.sum()
        total_h = h.sum()
        
        for feature in feature_indices:
            # 按特征值排序
            sorted_idx = np.argsort(X[:, feature])
            X_sorted = X[sorted_idx, feature]
            g_sorted = g[sorted_idx]
            h_sorted = h[sorted_idx]
            
            # 累积统计量
            sum_g_left, sum_h_left = 0.0, 0.0
            sum_g_right, sum_h_right = total_g, total_h
            
            for i in range(1, len(X_sorted)):
                sum_g_left += g_sorted[i-1]
                sum_g_right -= g_sorted[i-1]
                sum_h_left += h_sorted[i-1]
                sum_h_right -= h_sorted[i-1]
                
                # 跳过相同值的分裂点或者左子节点权重过小
                if X_sorted[i] == X_sorted[i-1] or sum_h_left < self.min_child_weight:
                    continue
                if sum_h_right < self.min_child_weight: break

                # 计算Loss reduction
                gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda) +
                        sum_g_right**2 / (sum_h_right + self.reg_lambda) -
                        total_g**2 / (total_h + self.reg_lambda))) - self.gamma / 2
                        
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (X_sorted[i-1] + X_sorted[i]) / 2
                    
        return best_gain, best_feature, best_threshold 
    
    def _sparse_split_find(self, X, g, h, feature_indices):
        """稀疏分裂寻找"""
        best_gain = 0.0
        total_g, total_h = g.sum(), h.sum()

        for feature in feature_indices:
            
            # exclude the missing values in X
            non_miss_indices = X[:, feature] != '?'
            X_non_miss = X[non_miss_indices, feature]
            g_non_miss = g[non_miss_indices]
            h_non_miss = h[non_miss_indices]
            
            if len(X_non_miss) == 0:
                continue 

            # 1.所有missing value在右侧，所以我们从左侧累积
            asc_sorted_idx = np.argsort(X_non_miss)
            X_asc_sorted = X_non_miss[asc_sorted_idx]
            g_sorted = g_non_miss[asc_sorted_idx]
            h_sorted = h_non_miss[asc_sorted_idx]

            sum_g_left, sum_h_left = 0.0, 0.0
            for i in range(1, len(X_asc_sorted)):
                sum_g_left += g_sorted[i-1]
                sum_h_left += h_sorted[i-1]
                sum_g_right = total_g - sum_g_left
                sum_h_right = total_h - sum_h_left
                
                # 检查最小子节点权重约束
                if sum_h_left < self.min_child_weight or sum_h_right < self.min_child_weight:
                    continue

                # 计算Loss reduction
                gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda) +
                        sum_g_right**2 / (sum_h_right + self.reg_lambda) -
                        total_g**2 / (total_h + self.reg_lambda))) - self.gamma / 2
                        
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (X_asc_sorted[i-1] + X_asc_sorted[i]) / 2

            # 2.所有missing value在左侧，所以我们从右侧累积
            desc_sorted_idx = asc_sorted_idx[::-1]
            X_desc_sorted = X_non_miss[desc_sorted_idx]
            g_desc_sorted = g_non_miss[desc_sorted_idx]
            h_desc_sorted = h_non_miss[desc_sorted_idx]

            sum_g_right, sum_h_right = 0.0, 0.0
            for i in range(1, len(X_desc_sorted)):
                sum_g_right += g_desc_sorted[i-1]
                sum_h_right += h_desc_sorted[i-1]
                sum_g_left = total_g - sum_g_right
                sum_h_left = total_h - sum_h_right
                
                # 检查最小子节点权重约束
                if sum_h_left < self.min_child_weight or sum_h_right < self.min_child_weight:
                    continue

                # 计算Loss reduction
                gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda) +
                        sum_g_right**2 / (sum_h_right + self.reg_lambda) -
                        total_g**2 / (total_h + self.reg_lambda))) - self.gamma / 2
                        
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (X_desc_sorted[i-1] + X_desc_sorted[i]) / 2

        return best_gain, best_feature, best_threshold

    def predict(self, X):
        """预测样本（确保X为numpy数组）"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(x, self.root) for x in X])
    
    def _predict(self, x, node):
        """递归遍历树"""
        if (node.best_gain == 0.0) or \
         (node.left is None and node.right is None) or \
         node.threshold is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)