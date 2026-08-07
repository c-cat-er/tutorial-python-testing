import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


# 假設這是專案中的蒸餾損失函數
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # 計算硬標籤交叉熵損失
    hard_loss = F.cross_entropy(student_logits, labels)
    # 計算軟標籤 KL 散度損失 (注意：KLDivLoss 的 input 需要是 log_softmax，target 是 softmax)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T**2)
    # 權重結合
    return alpha * hard_loss + (1 - alpha) * soft_loss


class TestDistillationLoss(unittest.TestCase):
    def setUp(self):
        # 模擬 Batch Size = 2, 類別數 = 4
        self.batch_size = 2
        self.num_classes = 4

        # 模擬學生與老師的 Logits 輸出
        self.student_logits = torch.tensor(
            [[2.0, 0.5, 0.1, -1.0], [0.2, 3.0, 0.5, 0.0]], dtype=torch.float32
        )
        self.teacher_logits = torch.tensor(
            [[2.5, 0.4, 0.0, -1.5], [0.1, 3.5, 0.6, -0.1]], dtype=torch.float32
        )
        self.labels = torch.tensor([0, 1], dtype=torch.long)

    def test_loss_is_valid_scalar(self):
        """測試損失函數是否能正確輸出純量，且不為 NaN 或 Inf"""
        loss = distillation_loss(self.student_logits, self.teacher_logits, self.labels)
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertFalse(torch.isnan(loss).any(), "蒸餾損失產出了 NaN！")
        self.assertFalse(torch.isinf(loss).any(), "蒸餾損失產出了 Inf！")

    def test_teacher_alignment(self):
        """測試當學生與老師的 Logits 完全一致時，軟標籤損失應該趨近於 0"""
        # 當兩者完全一樣時，KL 散度應為 0，整體損失只剩 alpha * hard_loss
        alpha = 0.0  # 設為 0 代表純看軟標籤損失
        loss = distillation_loss(
            self.student_logits, self.student_logits, self.labels, T=2.0, alpha=alpha
        )
        self.assertAlmostEqual(
            loss.item(), 0.0, places=5, msg="師生分佈一致時，KL 損失未歸零！"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
