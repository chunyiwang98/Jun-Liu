import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import shap
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib as mpl



mpl.rcParams['font.size'] = 10 
mpl.rcParams['font.family'] = 'Calibri'  


def set_random_seed(seed=995):
    random.seed(seed)
    np.random.seed(seed)


def evaluate_ensemble(y_true, proba_pred, threshold=0.5, dataset_name=""):
    y_pred = (proba_pred >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, proba_pred)

    print(f"{dataset_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    return accuracy, precision, recall, f1, roc_auc

set_random_seed(995)


train_data = pd.read_excel("train_data_clear1.xlsx").dropna()
test_data = pd.read_excel("test_data_clear1.xlsx").dropna()

y_train = train_data['group'].values
y_test = test_data['group'].values
feature_cols = [col for col in train_data.columns if col not in ['ID', 'group']]
X_train_full = train_data[feature_cols].values
X_test_full = test_data[feature_cols].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)


C_val = 0.9
n_selected_features = 60
lr = LogisticRegression(solver='liblinear', penalty='l1', C=C_val, random_state=995)
selector = RFE(lr, n_features_to_select=n_selected_features)
selector.fit(X_train_scaled, y_train)
coef_mask = selector.support_
X_train = X_train_scaled[:, coef_mask]
X_test = X_test_scaled[:, coef_mask]


catboost_params = {
    'iterations': 100,
    'depth': 10,
    'learning_rate': 0.69,
    'loss_function': 'Logloss',
    'cat_features': [],
    'random_seed': 995,
    'logging_level': 'Silent'
}

xgboost_params = {
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'learning_rate': 0.953,
    'n_estimators': 128,
    'max_depth': 6,
    'gamma': 0.02,
    'min_child_weight': 1.47,
    'reg_alpha': 0.11,
    'verbosity': 0,
    'random_state': 995
}

meta_model = GradientBoostingClassifier(n_estimators=164, learning_rate=0.6, max_depth=2,
                                        min_samples_split=0.73, random_state=995)

kf_outer = KFold(n_splits=5, shuffle=True, random_state=995)
meta_X, meta_y = [], []
ensemble_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

print("\n🚀 Start 5-Fold Cross Validation for Stacking (XGB + CAT => GBDT)\n")
for fold, (train_idx, val_idx) in enumerate(tqdm(kf_outer.split(X_train, y_train), desc="Cross Validation", total=5)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model_xgb = xgb.XGBClassifier(**xgboost_params)
    model_cat = CatBoostClassifier(**catboost_params)

    model_xgb.fit(X_tr, y_tr)
    model_cat.fit(X_tr, y_tr)

    val_proba_xgb = model_xgb.predict_proba(X_val)[:, 1]
    val_proba_cat = model_cat.predict_proba(X_val)[:, 1]

    val_meta_features = np.vstack((val_proba_xgb, val_proba_cat)).T
    meta_X.append(val_meta_features)
    meta_y.append(y_val)

    acc, prec, rec, f1, auc_score = evaluate_ensemble(y_val, val_meta_features[:, 0],
                                                      dataset_name=f"Fold {fold + 1} - Validation")
    ensemble_metrics['accuracy'].append(acc)
    ensemble_metrics['precision'].append(prec)
    ensemble_metrics['recall'].append(rec)
    ensemble_metrics['f1'].append(f1)
    ensemble_metrics['auc'].append(auc_score)

meta_X = np.vstack(meta_X)
meta_y = np.concatenate(meta_y)

meta_model.fit(meta_X, meta_y)

meta_pred_proba = meta_model.predict_proba(meta_X)[:, 1]
print("\n=== Meta Learner Stacking Summary (Train) ===")
evaluate_ensemble(meta_y, meta_pred_proba, dataset_name="Meta Learner (Train)")

print(f"\nAverage Accuracy: {np.mean(ensemble_metrics['accuracy']):.4f}")
print(f"Average Precision: {np.mean(ensemble_metrics['precision']):.4f}")
print(f"Average Recall: {np.mean(ensemble_metrics['recall']):.4f}")
print(f"Average F1-Score: {np.mean(ensemble_metrics['f1']):.4f}")
print(f"Average ROC-AUC: {np.mean(ensemble_metrics['auc']):.5f}")


final_model_xgb = xgb.XGBClassifier(**xgboost_params)
final_model_cat = CatBoostClassifier(**catboost_params)
final_model_xgb.fit(X_train, y_train)
final_model_cat.fit(X_train, y_train)

test_proba_xgb = final_model_xgb.predict_proba(X_test)[:, 1]
test_proba_cat = final_model_cat.predict_proba(X_test)[:, 1]
test_meta_input = np.vstack((test_proba_xgb, test_proba_cat)).T
test_proba_final = meta_model.predict_proba(test_meta_input)[:, 1]

print("\n=== Final Evaluation on Test Set (Stacking) ===")
evaluate_ensemble(y_test, test_proba_final, dataset_name="Test Set")


print("\n🎯 Computing SHAP values for the entire stacking model (based on raw features)...")


X_explain = X_train[:97]

def stacking_predict_proba(X):
    proba_xgb = final_model_xgb.predict_proba(X)[:, 1]
    proba_cat = final_model_cat.predict_proba(X)[:, 1]
    meta_input = np.vstack((proba_xgb, proba_cat)).T
    return meta_model.predict_proba(meta_input)[:, 1]


explainer = shap.KernelExplainer(stacking_predict_proba, X_explain)


shap_values = explainer.shap_values(X_explain)


selected_feature_names = [feature for feature, keep in zip(feature_cols, coef_mask) if keep]


X_explain_df = pd.DataFrame(X_explain, columns=selected_feature_names)


plt.xlim(-0.2, 0.2)
shap.summary_plot(shap_values, X_explain_df, plot_type="dot", max_display=10)

plt.xlim(-0.1, 0.1)
shap.summary_plot(shap_values, X_explain_df, plot_type="bar", max_display=10)




