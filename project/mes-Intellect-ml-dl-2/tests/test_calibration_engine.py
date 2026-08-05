import unittest

import numpy as np


# 假設這是您專案中的迭代校正演算法
def calibrate_iterative(prob_matrix, max_iter=5, eps=1e-10):
    # 防禦性檢查：確保輸入不是空矩陣
    if prob_matrix.size == 0:
        return prob_matrix

    calibrated_probs = prob_matrix.copy()
    for _ in range(max_iter):
        # 模擬內部 EM 或先驗調整迭代 (加上 eps 防止 log(0) 或除以 0)
        p_mean = np.mean(calibrated_probs, axis=0, keepdims=True) + eps
        calibrated_probs = calibrated_probs / p_mean

        # 重新歸一化 row sum = 1
        row_sums = np.sum(calibrated_probs, axis=1, keepdims=True) + eps
        calibrated_probs = calibrated_probs / row_sums

    return calibrated_probs


class TestCalibrationConvergence(unittest.TestCase):
    def test_extreme_zero_probabilities(self):
        """測試當包含極端逼近 0 的預測機率時，演算法是否能防禦 NaN"""
        # 模擬 3 筆晶圓數據，4 種缺陷類別。其中有非常極端的 0.0 機率
        extreme_probs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.9999, 0.0001, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=float,
        )

        calibrated = calibrate_iterative(extreme_probs, max_iter=5)

        # 斷言：輸出矩陣中絕對不能包含任何 NaN 或 無限大 (Inf)
        self.assertFalse(
            np.isnan(calibrated).any(), "迭代校正演算法產生了 NaN 錯誤值！"
        )
        self.assertFalse(
            np.isinf(calibrated).any(), "迭代校正演算法產生了 Inf 無限大值！"
        )

    def test_probability_distribution_properties(self):
        """測試校正後的矩陣是否仍保持機率性質（各類別總和為 1，且皆大於等於 0）"""
        random_probs = np.random.dirichlet(
            np.ones(4), size=10
        )  # 生成 10 筆標準機率分佈

        calibrated = calibrate_iterative(random_probs, max_iter=5)

        # 1. 斷言所有機率值都必須在合法的 [0, 1] 區間內
        self.assertTrue(
            (calibrated >= 0).all() and (calibrated <= 1).all(),
            "校正後機率值超出 [0,1] 範圍！",
        )

        # 2. 斷言每一列 (每片晶圓) 的類別機率加總必須精準等於 1 (容許浮點數微小誤差)
        row_sums = np.sum(calibrated, axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-5, err_msg="校正後的機率橫列加總不等於 1！"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
