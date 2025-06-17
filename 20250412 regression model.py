import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


file_path = 'E:/研三/MSA EEG/1225结果/20250219相关性分析/随访20250401.xlsx'
data = pd.read_excel(file_path)


data_clean = data.dropna() 
y_clean = data_clean.iloc[:, 6]  
X_clean = data_clean.iloc[:, 7:]  


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_clean)
y_scaled = scaler_y.fit_transform(y_clean.values.reshape(-1, 1)).flatten()


lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y_scaled)

selector = SelectFromModel(lasso, threshold="mean", max_features=40)
X_selected = selector.transform(X_scaled)
selected_features = X_clean.columns[selector.get_support()]
print("Selected Features:", selected_features)


best_r2 = -np.inf
best_seed = None
best_mse = None

k = 5
seed_range = [1794]


r2_scores = []
mse_scores = []

for seed in tqdm(seed_range):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    y_true_all = []
    y_pred_all = []

    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]


        model_xgb = XGBRegressor(
            n_estimators=268,
            max_depth=3,
            learning_rate=0.0579,
            colsample_bytree=0.75,
            subsample=0.86,
            reg_alpha=0.18,
            random_state=seed,
            tree_method='exact',
            predictor='cpu_predictor',
            enable_categorical=False,
            verbosity=0
        )

        model_rf = RandomForestRegressor(
            n_estimators=15,  
            max_depth=7,  
            random_state=seed,  
            n_jobs=-1  
        )


        model_xgb.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)


        xgb_pred = model_xgb.predict(X_test)
        rf_pred = model_rf.predict(X_test)


        stacked_features = np.column_stack((xgb_pred, rf_pred))


        meta_model = ElasticNet(alpha=0.001, l1_ratio=0.01)  # ElasticNet 参数可调整
        meta_model.fit(stacked_features, y_test)


        meta_pred = meta_model.predict(stacked_features)


        y_true_all.extend(scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten())
        y_pred_all.extend(scaler_y.inverse_transform(meta_pred.reshape(-1, 1)).flatten())


    current_r2 = r2_score(y_true_all, y_pred_all)
    current_mse = mean_squared_error(y_true_all, y_pred_all)


    r2_scores.append(current_r2)
    mse_scores.append(current_mse)


final_r2 = np.mean(r2_scores)
final_mse = np.mean(mse_scores)


print("\n✅ k fold results：")
print(f"Average R² Score: {final_r2:.4f}")
print(f"Average MSE:      {final_mse:.4f}")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


data_plot = pd.DataFrame({
    'True': y_true_all,  
    'Predicted': y_pred_all 
})


plt.figure(figsize=(8, 6), dpi=1200)
g = sns.JointGrid(data=data_plot, x="True", y="Predicted", height=10)


g.plot_joint(sns.scatterplot, color='#e6a56b', alpha=0.5)


sns.regplot(data=data_plot, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#e6a56b', label='Cross-validation Regression Line')


g.plot_marginals(sns.histplot, kde=False, element='bars', color='#e6a56b', alpha=0.5)

ax = g.ax_joint
ax.text(0.95, 0.1, f' $R^2$ = {final_r2:.3f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


ax.plot([data_plot['True'].min(), data_plot['True'].max()], [data_plot['True'].min(), data_plot['True'].max()], c="black", alpha=0.5, linestyle='--', label='x=y')
ax.legend()


plt.savefig("TrueFalse_StackedModel.pdf", format='pdf', bbox_inches='tight')
plt.show()



plt.figure(figsize=(6, 4), dpi=120)  # 调整画布大小和分辨率
residuals = data_plot['True'] - data_plot['Predicted']  # 计算残差


sns.scatterplot(x=data_plot['Predicted'], y=residuals, color='#e6a56b', alpha=0.5)

plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero Residual')

plt.title('Residual Plot', fontsize=12)  # 减小标题字体大小
plt.xlabel('Predicted Values', fontsize=10)  # 减小 x 轴标签字体大小
plt.ylabel('Residuals', fontsize=10)  # 减小 y 轴标签字体大小


plt.legend(fontsize=8)  # 减小图例字体大小


plt.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()

plt.savefig("Residual_Plot.pdf", format='pdf', bbox_inches='tight')
plt.show()

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define the function for stacking model prediction (already provided above)
def stacking_predict_proba(X):
    # Use the base learners' predictions
    xgb_pred = model_xgb.predict(X)
    rf_pred = model_rf.predict(X)

    # Stack base learners' predictions as new features
    stacked_features = np.column_stack((xgb_pred, rf_pred))

    # Use ElasticNet as the meta-model to make final predictions
    meta_pred = meta_model.predict(stacked_features)
    return meta_pred

# Select a subset of samples for SHAP explanation (you already defined X_selected and y_scaled)
X_explain = X_selected[:97]  # Selecting a subset of samples for explanation (to speed up computation)

# Create SHAP KernelExplainer object
explainer = shap.KernelExplainer(stacking_predict_proba, X_explain)

# Calculate SHAP values
shap_values = explainer.shap_values(X_explain)

# Convert X_explain to DataFrame for easy visualization
X_explain_df = pd.DataFrame(X_explain, columns=selected_features)

# Generate SHAP summary plot: dot plot
plt.figure(figsize=(10, 6), dpi=120)
shap.summary_plot(shap_values, X_explain_df, plot_type="dot", max_display=8)

# Customize the font settings
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10

# Generate SHAP summary plot: bar plot
plt.figure(figsize=(10, 6), dpi=120)
shap.summary_plot(shap_values, X_explain_df, plot_type="bar", max_display=8)

# Customize the font settings for bar plot as well
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10

# Save and display the plots
plt.savefig("SHAP_Summary_Plot.pdf", format='pdf', bbox_inches='tight')
plt.show()


