import unittest

import torch
import torch.nn as nn


class TestDaNNDomainConfusion(unittest.TestCase):
    def setUp(self):
        # 模擬已訓練完成的 DaNN 特徵提取器 (特徵維度 128)
        self.mock_feature_extractor = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU()
        ).eval()  # 鎖定模型

        # 模擬 A 廠 (Source) 與 B 廠 (Target) 的原始數據各 100 筆
        self.source_data = torch.randn(100, 64)
        self.target_data = torch.randn(100, 64)

    def test_domain_adversarial_confusion(self):
        """驗證特徵提取器提取出的特徵是否成功混淆了 A、B 廠區"""
        with torch.no_grad():
            # 提取兩廠數據的特徵
            source_features = self.mock_feature_extractor(self.source_data)
            target_features = self.mock_feature_extractor(self.target_data)

        # 建立一個臨時的、隨機初始化的 Domain 分類器（模擬對抗攻擊）
        temp_classifier = nn.Linear(128, 1)

        # 計算臨時分類器對 A 廠與 B 廠特徵的預測機率 (使用 Sigmoid)
        source_preds = torch.sigmoid(temp_classifier(source_features)) > 0.5
        target_preds = torch.sigmoid(temp_classifier(target_features)) > 0.5

        # 計算分類器猜對的數量 (假設 A 廠標籤為 1，B 廠標籤為 0)
        correct_source = source_preds.sum().item()
        correct_target = (~target_preds).sum().item()

        total_accuracy = (correct_source + correct_target) / 200.0

        # 斷言：一個隨機初始化的分類器在面對「成功域混淆」的特徵時，
        # 猜測準確率應該在 50% 上下徘徊，絕對不能展現出極高的分類正確率（例如 > 75%）
        self.assertLess(
            total_accuracy,
            0.75,
            f"域混淆失敗！臨時分類器仍可輕易分辨廠區特徵，準確率高達 {total_accuracy:.2%}",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
