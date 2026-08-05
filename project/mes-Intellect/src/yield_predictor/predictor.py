import pandas as pd
import shap
import xgboost as xgb


def predict_yield(mes_df, npi_df, wafer_features):
    keys = ["lot_id", "wafer_id"]
    npi_df = npi_df.merge(mes_df[keys], on=keys, how="inner")
    X = pd.concat(
        [
            mes_df.set_index(keys)[["severity"]],
            npi_df.set_index(keys)[["Value"]],
            pd.DataFrame(
                wafer_features, index=mes_df.set_index(keys).index
            ),  # 補齊 index
        ],
        axis=1,
    )
