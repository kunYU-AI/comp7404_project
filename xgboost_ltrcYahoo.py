import numpy as np
from xgboost import XGBRanker, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from xgboost import plot_importance

#DataSet
def parse_ltr_data(file_path):
    features = []
    labels = []
    qids = []
    feature_max_idx = 0  
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            labels.append(label)
            
            qid = int(parts[1].split(':')[1])
            qids.append(qid)
            
            feat_dict = {}
            for feat in parts[2:]:
                idx, val = feat.split(':')
                idx = int(idx)
                val = float(val)
                feat_dict[idx] = val
                if idx > feature_max_idx:
                    feature_max_idx = idx
            
            features.append(feat_dict)
    
    X = np.zeros((len(features), feature_max_idx + 1), dtype=np.float32)
    for i, feat_dict in enumerate(features):
        for idx, val in feat_dict.items():
            X[i, idx] = val
    
    return X, np.array(labels), np.array(qids)


X, y, qids = parse_ltr_data('./dataset/set1.train.txt')

unique_qids, group_sizes = np.unique(qids, return_counts=True)
groups = group_sizes.tolist()

train_qids = np.random.choice(
    unique_qids, 
    size=int(len(unique_qids) * 0.8), 
    replace=False
)

train_mask = np.isin(qids, train_qids)
X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]
qids_train, qids_val = qids[train_mask], qids[~train_mask]

_, train_group_sizes = np.unique(qids_train, return_counts=True)
_, val_group_sizes = np.unique(qids_val, return_counts=True)
train_groups = train_group_sizes.tolist()
val_groups = val_group_sizes.tolist()

#xgboost
model = XGBRanker(
    objective='rank:ndcg',      
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    group=train_groups,
    eval_set=[(X_val, y_val)],
    eval_group=[val_groups],
    verbose=True
)


y_pred_val = model.predict(X_val)
val_ndcg_scores = []
start = 0
for size in val_group_sizes:
    end = start + size
    true_labels = y_val[start:end]
    pred_scores = y_pred_val[start:end]
    val_ndcg_scores.append(ndcg_score([true_labels], [pred_scores], k=5))
    start = end

print(f"\nValidation NDCG@5: {np.mean(val_ndcg_scores):.4f}")

#Plot
plot_importance(model, max_num_features=20)
plt.show()