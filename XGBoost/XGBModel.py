import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Tool import var_contribution_filter,train_single_xgb
from Tool import plot_feature_bar,plot_true_vs_pred,plot_shap_for_model

warnings.filterwarnings("ignore")

# 参数设置
RANDOM_STATE = 42
TEST_SIZE = 0.3
OUTPUT_DIR = "XGB"
os.makedirs(OUTPUT_DIR, exist_ok=True)
USE_HYPERPARAMETER_SEARCH = True
N_ITER_SEARCH = 50
VAR_CONTRIB_THRESHOLD = 0.01
ENABLE_VAR_FILTER = True   # True=启用方差筛选，False=使用全部特征

# 数据加载与预处理
data_path = "./data.csv"
df = pd.read_csv(data_path, encoding="gbk")
output_cols = ["Y"]
all_q_columns = [col for col in df.columns if col.startswith("Q")]
input_cols = [col for col in all_q_columns if col not in output_cols]
print(f"输入特征列表（共{len(input_cols)}个）：{input_cols}")
X_raw = df[input_cols].copy().apply(pd.to_numeric, errors="coerce")
Y = df[output_cols].copy().apply(pd.to_numeric, errors="coerce")
# 缺失值填充
X_raw = X_raw.fillna(X_raw.median())
Y = Y.fillna(Y.median())
# 特征筛选
if ENABLE_VAR_FILTER:
    X, feature_names, var_contrib_df = var_contribution_filter(X_raw, VAR_CONTRIB_THRESHOLD)
    var_contrib_df.to_csv(os.path.join(OUTPUT_DIR, "variance_contribution.csv"), index=False)
    print(f"【启用方差筛选】原始特征：{len(input_cols)}，筛选后特征：{len(feature_names)}")
else:
    X = X_raw.values.astype(np.float32)
    feature_names = X_raw.columns.tolist()
    print(f"【关闭方差筛选】使用全部特征，数量：{len(feature_names)}")
Y = Y.values.astype(np.float32)
target_names = output_cols

# 训练/测试集拆分
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

# 模型训练
models = []
train_fit_actuals, train_fit_preds = [], []
val_actuals, val_preds = [], []
test_actuals, test_preds = [], []
metrics_list = []

for i, target in enumerate(target_names):
    model, y_train_fit_actual, y_train_fit_pred, y_val_actual, y_val_pred, y_test_pred, metrics = train_single_xgb(
        X_train, Y_train[:, i], X_test, Y_test[:, i], target,use_tuning=USE_HYPERPARAMETER_SEARCH,random_state=RANDOM_STATE)

    models.append(model)
    train_fit_actuals.append(y_train_fit_actual)
    train_fit_preds.append(y_train_fit_pred)
    val_actuals.append(y_val_actual)
    val_preds.append(y_val_pred)
    test_actuals.append(Y_test[:, i])
    test_preds.append(y_test_pred)
    metrics_list.append(metrics)

# 转为列状数组
train_fit_actuals = np.column_stack(train_fit_actuals)
train_fit_preds = np.column_stack(train_fit_preds)
val_actuals = np.column_stack(val_actuals)
val_preds = np.column_stack(val_preds)
test_actuals = np.column_stack(test_actuals)
test_preds = np.column_stack(test_preds)

# 保存指标与预测结果
all_metrics = pd.concat(metrics_list, ignore_index=True)
all_metrics.to_csv(os.path.join(OUTPUT_DIR, "xgb_metrics.csv"), index=False)
print("\n==== Final Metrics ====")
print(all_metrics)

# 保存三层预测CSV
def save_predictions(actuals, preds, name, output_dir):
    result = pd.DataFrame(actuals, columns=[f"{t}_true" for t in target_names])
    for i, t in enumerate(target_names):
        result[f"{t}_pred"] = preds[:, i]
    result.to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)

save_predictions(train_fit_actuals, train_fit_preds, "train_fit", OUTPUT_DIR)
save_predictions(val_actuals, val_preds, "val", OUTPUT_DIR)
save_predictions(test_actuals, test_preds, "test", OUTPUT_DIR)

# 特征重要性
feature_importance_list = []
for i, target in enumerate(target_names):
    booster = models[i].get_booster()
    fi = booster.get_score(importance_type="gain")
    fi_raw = np.array([fi.get(f"f{idx}", 0.0) for idx in range(len(feature_names))])
    total_gain = np.sum(fi_raw)
    fi_norm = fi_raw / total_gain if total_gain > 1e-8 else fi_raw

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": fi_norm,
        "target": target
    })
    feature_importance_list.append(fi_df)

feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

for target in target_names:
    tmp = feature_importance_df[feature_importance_df["target"] == target]
    plot_feature_bar(tmp, title=f"Top 9 Importance for {target}",
                     filename=f"{target}_top9_importance.png", output_dir=OUTPUT_DIR)
    plot_feature_bar(tmp, title=f"Bottom 9 Importance for {target}",
                     filename=f"{target}_bottom9_importance.png", output_dir=OUTPUT_DIR, ascending=True)

# SHAP分析与真值-预测图
X_shap = X_test[:min(200, X_test.shape[0])]
for i, target in enumerate(target_names):
    print(f"Computing SHAP for {target} ...")
    plot_shap_for_model(models[i], X_shap, target, feature_names, OUTPUT_DIR, top_k=9)
    plot_true_vs_pred(test_actuals[:, i], test_preds[:, i], target, OUTPUT_DIR)

print("\n==== All Finished ====")
print(f"所有结果、图片已保存在：{OUTPUT_DIR}")