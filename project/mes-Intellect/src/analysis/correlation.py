import pandas as pd


def correlate_probe_final_test(probe_df, final_df):
    return probe_df.merge(final_df, on="lot_id", how="inner").corr()
