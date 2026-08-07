import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def calculate_spc_cpk(data, usl, lsl):
    """計算統計製程管制 (SPC) 的關鍵品質指標 Cpk"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    return float(min(cpu, cpl))


def run_anova_test(group_a, group_b, group_c):
    """製程機台變更或 NPI 驗證時常用的方差分析 (ANOVA)"""
    f_stat, p_val = stats.f_oneway(group_a, group_b, group_c)
    return {"f_statistic": float(f_stat), "p_value": float(p_val)}


def run_advanced_regression_analysis(X_features: pd.DataFrame, y_yield: pd.Series):
    """
    使用 statsmodels 執行多元線性迴歸
    量化各電性參數 (X) 對晶圓良率 (y) 的獨立貢獻度與 p-value 顯著性
    """
    # statsmodels 必須手動加入常數項 (截距 Intercept)
    X_with_constant = sm.add_constant(X_features)

    # 建立並擬合 OLS 模型
    model = sm.OLS(y_yield, X_with_constant)
    results = model.fit()

    # 提取 YIE（良率整合工程師）最看重的參數
    return {
        "r_squared": float(results.rsquared),  # 模型解釋力
        "adjusted_r_squared": float(results.rsquared_adj),
        "coefficients": results.params.to_dict(),  # 每個參數的權重
        "p_values": results.pvalues.to_dict(),  # 每個參數的統計顯著度
    }
