import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb

#DataSet
data = pd.read_csv('./dataset/higgsboson_training.csv')

print("Dimension:", data.shape)
print("\nFirst 3 row:")
print(data.head(3))
print("\nLabels:")
print(data['Label'].value_counts())

#Data Processing
features = data.drop(['EventId', 'Label', 'Weight'], axis=1)
labels = data['Label'].map({'s': 1, 'b': 0})  
sample_weights = data['Weight'].values  

#NaN
features = features.fillna(features.median()) 


X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
    features, labels, sample_weights, 
    test_size=0.2, random_state=42, stratify=labels
)

#Xgboost model
model = xgb.XGBClassifier(
    objective='binary:logistic',  
    n_estimators=650,           
    max_depth=25,                
    learning_rate=0.5,          
    subsample=0.7,             
    colsample_bytree=0.8,       
    reg_lambda=0.7,             
    scale_pos_weight=np.sum(labels == 0) / np.sum(labels == 1) 
)

#Fit
model.fit(
    X_train, y_train,
    sample_weight=weights_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

#Evaluation
y_pred_proba = model.predict_proba(X_val)[:, 1]

print("AUC-ROC:", roc_auc_score(y_val, y_pred_proba))
print("Accuracy Score:", accuracy_score(y_val, (y_pred_proba > 0.3).astype(int)))

#Confusion_matrix
cm = confusion_matrix(y_val, (y_pred_proba > 0.5).astype(int))
print("\nConfusion Matrix:")
print(cm)

#Plot
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=15)
plt.show()

# model.save_model('higgsboson_xgboost.model')