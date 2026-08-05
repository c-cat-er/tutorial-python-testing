import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.common.utils import prepare_feature_matrix


def predict_yield(mes_df, npi_df, wafer_features):
    # 1. 直接調用修正後的統一特徵工程（此時 X, y 結構絕對完美一維）
    X, y = prepare_feature_matrix(mes_df, npi_df, wafer_features)

    # 2. 執行訓練與預測
    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    # 3. SHAP 解釋
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(X.columns, importance.tolist()))

    return {
        "predicted_yield": float(preds.mean()),
        "top_factors": sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        ),
    }
