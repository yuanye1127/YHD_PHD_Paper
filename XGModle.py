import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================
# 1. 配置参数
# =========================
# 随机种子，保证每次划分数据和训练结果尽量一致
RANDOM_STATE = 42
# 测试集比例
TEST_SIZE = 0.2
# 结果保存目录
OUTPUT_DIR = "xgb_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 是否进行超参数搜索
USE_HYPERPARAMETER_SEARCH = True
# 搜索方式：random 更快；grid 更全面但更慢
SEARCH_METHOD = "random"
# 随机搜索次数
N_ITER_SEARCH = 15

# =========================
# 2. 读取真实 CSV 数据
# =========================
# 真实数据路径，换成你的文件名
data_path = "0522.csv"
# 读取 CSV 文件
# 如果 gbk 报错，可以改成 utf-8-sig
df = pd.read_csv(data_path, encoding="gbk")
# 指定输入输出列
input_cols = [f"Q{i}" for i in range(1, 51)]   # Q1 ~ Q50
output_cols = [f"Q{i}" for i in range(55, 58)]  # Q55 ~ Q57
# 检查列是否存在
missing_inputs = [c for c in input_cols if c not in df.columns]
missing_outputs = [c for c in output_cols if c not in df.columns]
if missing_inputs:
    raise ValueError(f"缺少输入列: {missing_inputs}")
if missing_outputs:
    raise ValueError(f"缺少输出列: {missing_outputs}")
# 提取输入输出
X = df[input_cols].copy()
Y = df[output_cols].copy()
# 转数值
X = X.apply(pd.to_numeric, errors="coerce")
Y = Y.apply(pd.to_numeric, errors="coerce")
# 缺失值处理
X = X.fillna(X.mean())
Y = Y.fillna(Y.mean())
# 转为 numpy
X = X.values.astype(np.float32)
Y = Y.values.astype(np.float32)

feature_names = input_cols
target_names = output_cols

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# 3. 划分训练测试集 + 标准化
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
#
# X_train = x_scaler.fit_transform(X_train)
# X_test = x_scaler.transform(X_test)
#
# Y_train = y_scaler.fit_transform(Y_train)
# Y_test = y_scaler.transform(Y_test)

# =========================
# 4. 超参数搜索
# =========================
def tune_xgb_params(X_train, y_train):
    # 基础 XGBoost 回归模型
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    # 候选参数范围
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],     # 树的数量
        "max_depth": [3, 4, 5, 6],                      # 每棵树的最大深度
        "learning_rate": [0.01, 0.05, 0.1],            # 学习率，越小越稳但越慢
        "subsample": [0.7, 0.8, 1.0],                   # 每棵树随机采样多少样本
        "colsample_bytree": [0.7, 0.8, 1.0],            # 每棵树随机采样多少特征
        "min_child_weight": [1, 3, 5],                  # 子节点最小样本权重
        "gamma": [0, 0.1, 0.2],                         # 分裂所需最小损失下降
        "reg_alpha": [0, 0.01, 0.1],                    # L1 正则
        "reg_lambda": [1, 1.5, 2]                       # L2 正则
    }
    # 选择网格搜索或随机搜索
    if SEARCH_METHOD.lower() == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=N_ITER_SEARCH,
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE
        )
    # 执行搜索
    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)
    print("Best Score:", search.best_score_)

    return search.best_params_

# =========================
# 5. 训练单个输出模型
# =========================
def train_single_xgb(X_train, y_train, X_test, y_test, target_name, use_tuning=True):
    print(f"\n===== Training model for {target_name} =====")
    # 如果启用调参，就先搜索最佳参数
    if use_tuning:
        best_params = tune_xgb_params(X_train, y_train)
    else:
        # 如果不调参，就用默认经验参数
        best_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1
        }
    # 用最佳参数构建最终模型
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params
    )
    # 拟合模型
    model.fit(X_train, y_train)
    # 训练集预测
    y_train_pred = model.predict(X_train)
    # 测试集预测
    y_test_pred = model.predict(X_test)
    # 训练集评价指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    # 测试集评价指标
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    # 打包成 DataFrame，方便保存
    metrics = pd.DataFrame([{
        "target": target_name,
        "Train_MSE": train_mse,
        "Train_RMSE": train_rmse,
        "Train_MAE": train_mae,
        "Train_R2": train_r2,
        "Test_MSE": test_mse,
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2
    }])

    return model, y_train_pred, y_test_pred, metrics

# =========================
# 6. 训练所有输出
# =========================
models = []
train_preds = []
test_preds = []
metrics_list = []

for i, target in enumerate(target_names):
    model, y_train_pred, y_test_pred, metrics = train_single_xgb(
        X_train, Y_train[:, i], X_test, Y_test[:, i], target,
        use_tuning=USE_HYPERPARAMETER_SEARCH
    )
    models.append(model)
    train_preds.append(y_train_pred)
    test_preds.append(y_test_pred)
    metrics_list.append(metrics)

train_preds = np.column_stack(train_preds)
test_preds = np.column_stack(test_preds)
all_metrics = pd.concat(metrics_list, ignore_index=True)

print("\n==== Metrics ====")
print(all_metrics)
all_metrics.to_csv(os.path.join(OUTPUT_DIR, "xgb_metrics.csv"), index=False)

# =========================
# 7. 保存预测结果
# =========================
train_result = pd.DataFrame(Y_train, columns=[f"{t}_true" for t in target_names])
for i, t in enumerate(target_names):
    train_result[f"{t}_pred"] = train_preds[:, i]

test_result = pd.DataFrame(Y_test, columns=[f"{t}_true" for t in target_names])
for i, t in enumerate(target_names):
    test_result[f"{t}_pred"] = test_preds[:, i]

train_result.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)
test_result.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# =========================
# 8. 保存特征重要性
# =========================
feature_importance_list = []

for i, target in enumerate(target_names):
    fi = models[i].feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": fi,
        "target": target
    })
    feature_importance_list.append(fi_df)

feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

# =========================
# 9. 画 top15 / bottom15 特征图
# =========================
def plot_feature_bar(df, title, filename, feature_col="feature", value_col="importance", top_n=15, ascending=False):
    df = df.sort_values(by=value_col, ascending=ascending).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=value_col, y=feature_col, color="steelblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

for target in target_names:
    tmp = feature_importance_df[feature_importance_df["target"] == target]

    # top 15
    plot_feature_bar(
        tmp,
        title=f"Top 15 Feature Importance for {target}",
        filename=f"{target}_top15_importance.png",
        top_n=15,
        ascending=False
    )

    # bottom 15
    plot_feature_bar(
        tmp,
        title=f"Bottom 15 Feature Importance for {target}",
        filename=f"{target}_bottom15_importance.png",
        top_n=15,
        ascending=True
    )

# =========================
# 10. SHAP 分析
# =========================
def plot_shap_for_model(model, X_sample, target_name, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # summary dot plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.show()

# 用测试集前 200 条做 SHAP，加快速度
X_shap = X_test[:min(200, X_test.shape[0])]

for i, target in enumerate(target_names):
    print(f"Computing SHAP for {target} ...")
    plot_shap_for_model(models[i], X_shap, target, feature_names)

# =========================
# 11. 真值 vs 预测值图
# =========================
def plot_true_vs_pred(y_true, y_pred, target_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted - {target_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_true_vs_pred.png"), dpi=300)
    plt.show()

for i, target in enumerate(target_names):
    plot_true_vs_pred(Y_test[:, i], test_preds[:, i], target)

# =========================
# 12. 完成
# =========================
print("\nAll done!")
print(f"Results saved in: {OUTPUT_DIR}")
