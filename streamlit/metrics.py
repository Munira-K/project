import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from data import load_data

data = load_data()

le = LabelEncoder()
data['satisfaction'] = le.fit_transform(data['satisfaction'])
data['Class'] = le.fit_transform(data['Class'])
data['Type of Travel'] = le.fit_transform(data['Type of Travel'])
data['Customer Type'] = le.fit_transform(data['Customer Type'])


X = data[[
    'Inflight entertainment',
    'Seat comfort',
    'On-board service',
    'Cleanliness',
    'Leg room service',
    'Inflight wifi service',
    'Baggage handling',
    'Checkin service',
    'Food and drink',
    'Type of Travel']]
y = data['satisfaction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logistic = LogisticRegression(max_iter=565)
logistic.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


des_tree = DecisionTreeClassifier(max_depth=5)
des_tree.fit(X_train, y_train)

lgbm_model = LGBMClassifier(verbose=-200)
lgbm_model.fit(X_train, y_train)


y_proba_knn_train = knn.predict_proba(X_train)[:, 1]
y_proba_knn_test = knn.predict_proba(X_test)[:, 1]

y_proba_log_train = logistic.predict_proba(X_train)[:, 1]
y_proba_log_test = logistic.predict_proba(X_test)[:, 1]

y_proba_tree_train = des_tree.predict_proba(X_train)[:, 1]
y_proba_tree_test = des_tree.predict_proba(X_test)[:, 1]

y_proba_lgbm_train = lgbm_model.predict_proba(X_train)[:, 1]
y_proba_lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]

roc_auc_test_knn = roc_auc_score(y_test, y_proba_knn_test)
roc_auc_train_knn = roc_auc_score(y_train, y_proba_knn_train)

roc_auc_test_log = roc_auc_score(y_test, y_proba_log_test)
roc_auc_train_log = roc_auc_score(y_train, y_proba_log_train)

roc_auc_test_tree = roc_auc_score(y_test, y_proba_tree_test)
roc_auc_train_tree = roc_auc_score(y_train, y_proba_tree_train)

roc_auc_test_lgbm = roc_auc_score(y_test, y_proba_lgbm_test) 
roc_auc_train_lgbm = roc_auc_score(y_train, y_proba_lgbm_train)

logistic_accuracy_train = accuracy_score(y_train, logistic.predict(X_train))
logistic_recall_train = recall_score(y_train, logistic.predict(X_train))
logistic_accuracy_test = accuracy_score(y_test, logistic.predict(X_test))
logistic_recall_test = recall_score(y_test, logistic.predict(X_test))

knn_accuracy_train = accuracy_score(y_train, knn.predict(X_train))
knn_recall_train = recall_score(y_train, knn.predict(X_train))
knn_accuracy_test = accuracy_score(y_test, knn.predict(X_test))
knn_recall_test = recall_score(y_test, knn.predict(X_test))

des_tree_accuracy_train = accuracy_score(y_train, des_tree.predict(X_train))
des_tree_recall_train = recall_score(y_train, des_tree.predict(X_train))
des_tree_accuracy_test = accuracy_score(y_test, des_tree.predict(X_test))
des_tree_recall_test = recall_score(y_test, des_tree.predict(X_test))

lgbm_accuracy_train = accuracy_score(y_train, lgbm_model.predict(X_train))
lgbm_recall_train = recall_score(y_train, lgbm_model.predict(X_train))
lgbm_accuracy_test = accuracy_score(y_test, lgbm_model.predict(X_test))
lgbm_recall_test = recall_score(y_test, lgbm_model.predict(X_test))

metrics_dict = {
    'Model': ['Logistic Regression', 'K-Nearest Neighbors (KNN)', 'Decision Tree', 'LightGBM (LGBM)'],
    'Accuracy (Test)': [logistic_accuracy_test, knn_accuracy_test, des_tree_accuracy_test, lgbm_accuracy_test],
    'Accuracy (Train)': [logistic_accuracy_train, knn_accuracy_train, des_tree_accuracy_train, lgbm_accuracy_train],
    'Recall (Test)': [logistic_recall_test, knn_recall_test, des_tree_recall_test, lgbm_recall_test],
    'Recall (Train)': [logistic_recall_train, knn_recall_train, des_tree_recall_train, lgbm_recall_train],
    'ROC AUC (Test)': [roc_auc_test_log, roc_auc_test_knn, roc_auc_test_tree, roc_auc_test_lgbm],
    'ROC AUC (Train)': [roc_auc_train_log, roc_auc_train_knn, roc_auc_train_tree, roc_auc_train_lgbm]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df = metrics_df.round(2)

st.dataframe(metrics_df)

cv_scores_knn = cross_val_score(knn, X, y, cv=5, scoring='recall')
cv_scores_knn_mean = cv_scores_knn.mean().round(2)
cv_scores_knn_std = cv_scores_knn.std().round(4)
cv_scores_log = cross_val_score(logistic, X, y, cv=5, scoring='recall')
cv_scores_log_mean = cv_scores_log.mean().round(2)
cv_scores_log_std = cv_scores_log.std().round(4)
cv_score_tree = cross_val_score(des_tree, X, y, cv=5, scoring='recall')
cv_scores_tree_mean = cv_score_tree.mean().round(2)
cv_scores_tree_std = cv_score_tree.std().round(4)
cv_scores_lgbm = cross_val_score(lgbm_model, X, y, cv=5, scoring='recall')
cv_scores_lgbm_mean = cv_scores_lgbm.mean().round(2)
cv_scores_lgbm_std = cv_scores_lgbm.std().round(4)
results = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Decision Tree', 'LGBM'],
    'Mean Recall': [cv_scores_knn_mean, cv_scores_log_mean, cv_scores_tree_mean, cv_scores_lgbm_mean],
    'Recall Std Dev': [cv_scores_knn_std, cv_scores_log_std, cv_scores_tree_std, cv_scores_lgbm_std]
})

st.dataframe(results)
