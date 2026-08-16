from pathlib import Path


def xlsx_to_csv(xlsx_path: str, csv_path: str = None, encoding="utf-8-sig"):
    """
    将xlsx文件转为csv
    :param xlsx_path: 输入xlsx文件路径
    :param csv_path: 输出csv路径，不传则自动同名替换后缀
    :param encoding: csv编码，utf-8-sig保证Excel打开不乱码
    """
    xlsx_file = Path(xlsx_path)
    if not xlsx_file.exists():
        print(f"错误：文件 {xlsx_path} 不存在！")
        return

    # 自动生成csv文件名
    if csv_path is None:
        csv_path = str(xlsx_file.with_suffix(".csv"))

    try:
        # 读取第一张工作表
        df = pd.read_excel(xlsx_path)
        # 保存csv，不输出pandas默认索引列
        df.to_csv(csv_path, index=False, encoding=encoding)
        print(f"转换成功！\n源文件：{xlsx_path}\n输出文件：{csv_path}")
    except Exception as e:
        print(f"转换失败：{str(e)}")


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")

# 特征筛选函数
def var_contribution_filter(X_df, threshold):
    """
    基于方差贡献率筛选特征
    """
    feature_var = X_df.var(axis=0)
    total_var = feature_var.sum()
    var_contribution = feature_var / total_var
    keep_mask = var_contribution >= threshold
    X_filtered = X_df.loc[:, keep_mask]
    keep_features = X_filtered.columns.tolist()
    contrib_df = pd.DataFrame({
        "feature": feature_var.index,
        "variance": feature_var.values,
        "var_contribution": var_contribution.values
    })
    return X_filtered.values.astype(np.float32), keep_features, contrib_df

# 超参数调优函数
def tune_xgb_params(X_train, y_train, n_iter_search=50, random_state=42):
    """
    使用随机搜索 + 重复交叉验证寻找最优超参数
    """
    param_grid = {
        "n_estimators": [500, 600, 650],
        "max_depth": [1, 2, 3],
        "learning_rate": [0.01, 0.02, 0.03],
        "subsample": [0.65, 0.7, 0.8],
        "colsample_bytree": [0.75, 0.85, 0.9],
        "min_child_weight": [6, 9, 12, 16],
        "gamma": [0.5, 1, 1.5],
        "reg_alpha": [1, 2, 4],
        "reg_lambda": [6, 8, 12]
    }

    base_model = xgb.XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=random_state
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )

    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)
    print("Best CV Score:", search.best_score_)
    return search.best_params_

# 模型训练函数
def train_single_xgb(X_train, y_train, X_test, y_test, target_name,use_tuning=True, random_state=42):
    print(f"\n===== Training model for {target_name} =====")

    # 从训练集中拆分验证集
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )

    # 超参数调优或固定参数
    if use_tuning:
        best_params = tune_xgb_params(X_train, y_train, random_state=random_state)
    else:
        best_params = {
            "subsample": 0.7,
            "reg_lambda": 12,
            "reg_alpha": 4,
            "n_estimators": 500,
            "min_child_weight": 12,
            "max_depth": 3,
            "learning_rate": 0.02,
            "gamma": 2,
            "colsample_bytree": 0.75
        }

    # 建立模型
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"]
    )

    # 训练，使用验证集做早停。
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    # 三层评估
    y_train_fit_pred = model.predict(X_train_fit)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train_fit, y_train_fit_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    metrics = pd.DataFrame([{
        "target": target_name,
        "Train_Fit_R2": r2_score(y_train_fit, y_train_fit_pred),
        "Val_R2": r2_score(y_val, y_val_pred),
        "Test_R2": r2_score(y_test, y_test_pred),
        "Train_Fit_RMSE": np.sqrt(train_mse),
        "Val_RMSE": np.sqrt(val_mse),
        "Test_RMSE": np.sqrt(test_mse)
    }])

    return model, y_train_fit, y_train_fit_pred, y_val, y_val_pred, y_test_pred, metrics

# 绘图函数
def plot_feature_bar(df, title, filename, output_dir, top_n=9, ascending=False):
    """绘制特征重要性条形图"""
    df = df.sort_values(by="importance", ascending=ascending).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature", color="steelblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=900)
    plt.show()

def plot_true_vs_pred(y_true, y_pred, target_name, output_dir):
    """绘制真值-预测散点图"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Test Set: True vs Predicted - {target_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_true_vs_pred.png"), dpi=900)
    plt.show()

# SHAP解释函数
def plot_shap_for_model(model, X_sample, target_name, feat_names, output_dir, top_k=9):
    """
    生成SHAP汇总图、条形图、组合图及Top特征的依赖图
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 组合图（散点+条形）
    fig = plt.figure(figsize=(11, 20))
    ax1 = plt.gca()
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False)

    ytick_labels = [text.get_text() for text in ax1.get_yticklabels()]
    n_feat = len(ytick_labels)
    y_pos = np.arange(n_feat)

    abs_shap_mean = np.mean(np.abs(shap_values), axis=0)
    name2imp = dict(zip(feat_names, abs_shap_mean))
    sorted_importance = np.array([name2imp[name] for name in ytick_labels])

    ax2 = ax1.twiny()
    ax2.barh(y=y_pos, width=sorted_importance, height=0.62, color="#87ceeb", alpha=0.42, zorder=1)
    ax2.set_xlabel("Mean |SHAP| (Feature Importance)")
    ax2.set_xlim(0, np.max(sorted_importance)*1.05)
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)

    plt.subplots_adjust(right=0.92)
    plt.savefig(os.path.join(output_dir, f"{target_name}_shap_combined.png"), dpi=900, bbox_inches="tight")
    plt.show()
    plt.close()

    # 标准汇总图
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_shap_summary.png"), dpi=900, bbox_inches="tight")
    plt.show()

    # 条形图
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_name}_shap_bar.png"), dpi=900, bbox_inches="tight")
    plt.show()

    # 依赖图（Top K）
    shap_importance = np.mean(np.abs(shap_values), axis=0)
    top_feature_idx = np.argsort(-shap_importance)[:top_k]

    for idx in top_feature_idx:
        feat_name = feat_names[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_values,
            features=X_sample,
            feature_names=feat_names,
            show=False
        )
        ax = plt.gca()
        scatter_points = ax.collections[0].get_offsets()
        x_raw = scatter_points[:, 0]
        y_raw = scatter_points[:, 1]

        lowess_result = sm.nonparametric.lowess(y_raw, x_raw, frac=0.6)
        fit_x = lowess_result[:, 0]
        fit_y = lowess_result[:, 1]

        ax.plot(fit_x, fit_y, color="red", lw=2.2, label="LOWESS Fit Curve")
        ax.legend(loc="best")
        ax.set_title(f"{target_name} | SHAP Dependence Plot: {feat_name}")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{target_name}_dependence_{feat_name}.png")
        plt.savefig(save_path, dpi=900, bbox_inches="tight")
        plt.show()
        plt.close()