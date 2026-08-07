import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# 晶圓圖專用影像增強轉換 (Data Augmentation)
def get_wafer_transforms(img_size=128):
    # 訓練集：幾何隨機變形以防止過擬合
    train_tfm = transforms.Compose(
        [
            transforms.ToPILImage(),  # 將 2D NumPy 矩陣轉為 PIL 以利增強
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # 新增：晶圓垂直翻轉亦具物理合理性
            transforms.RandomRotation(30),  # 放大至 30 度，適應更多產線漂移缺陷
            transforms.ToTensor(),  # 自動縮放像素至 [0.0, 1.0] 區間
        ]
    )

    # 驗證/測試/偽標籤預測集：嚴格固定尺寸，排除隨機性干擾
    test_tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tfm, test_tfm


# 晶圓實體資料集 (承接 NumPy 2D 矩陣)
class WaferImageDataset(Dataset):
    def __init__(self, matrices: list, labels: list = None, transform=None):
        self.matrices = matrices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        # 將原始 2D Bin Map 複製為 3 通道 (RGB)，以符合 ResNet18 的輸入格式要求
        matrix = self.matrices[idx]
        if matrix.ndim == 2:
            matrix = np.stack([matrix] * 3, axis=-1).astype(np.uint8)

        if self.transform:
            img_tensor = self.transform(matrix)
        else:
            img_tensor = torch.from_numpy(matrix).float().permute(2, 0, 1)

        if self.labels is not None:
            return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, torch.tensor(
            0, dtype=torch.long
        )  # 模擬筆記中的 DatasetFolder 假標籤


# 半監督雙視圖橋樑資料集 (完整複用筆記黑科技)
class PseudoDataset(Dataset):
    def __init__(self, subset_images, labels):
        self.subset_images = subset_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, _ = self.subset_images[idx]  # 觸發 unlabeled_set 的 train_tfm 強增強訓練
        label = self.labels[idx]
        return img, label


# 晶圓缺陷 ResNet18 分類器
class WaferResNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 載入經典 ResNet18 架構，依據說明不使用預訓練權重
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        # 重構 Fine-tuning Head，將全連接層輸出修改為 4 類缺陷
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


# 專案六新加入：無監督異常檢測模型群 (對齊專案架構)
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),  # 將輸出層限制在 [-1, 1] 區間
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class WaferVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Conv2d(24, 48, 4, stride=2, padding=1)  # Mu
        self.enc_out_2 = nn.Conv2d(24, 48, 4, stride=2, padding=1)  # LogVar

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)  # 重參數化技巧
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x, reduction="none").sum(dim=[1, 2, 3]).mean()
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]).mean()
    return mse + kld


# 異常檢測專用晶圓影像矩陣轉換資料集
class WaferAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, matrices: list, img_size=64):
        self.data = []
        for m in matrices:
            # 傳入的確定是 2D 矩陣，先轉成 Tensor 並擴展為 3 通道 (3, H, W)
            t = torch.from_numpy(m).float()
            if t.ndim == 2:
                resized = t.unsqueeze(0).repeat(3, 1, 1)
            else:
                resized = t.permute(2, 0, 1) if t.shape[-1] == 3 else t

            # 安全進行空間插值，精準對齊為 [3, img_size, img_size]
            resized = F.interpolate(
                resized.unsqueeze(0), size=(img_size, img_size), mode="nearest"
            ).squeeze(0)

            self.data.append(resized)

        self.data = torch.stack(self.data, dim=0)
        max_val = self.data.max() if self.data.max() > 0 else 1.0
        self.data = 2.0 * (self.data / max_val) - 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


import numpy as np
import torch
import torch.nn as nn


# 專案七新加入：領域自適應 (DaNN) 模型組件群
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        # 核心複用筆記優化：精準防禦當 Batch=1 線上推理時的維度坍塌崩潰
        return self.conv(x).squeeze(-1).squeeze(-1)


class LabelPredictor(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),  # 對齊半導體的 4 類缺陷
        )

    def forward(self, h):
        return self.layer(h)


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),  # 二分類器：辨識是來自 Factory A (1) 還是 Factory B (0)
        )

    def forward(self, h):
        return self.layer(h)


# 核心複用筆記強基線黑科技：盲測期先驗偏移迭代式校正技術
def calibrate_iterative(logits, max_iter=5, eps=1e-8):
    """
    量化各類別在測試集中的分佈，施加 log 懲罰扣分，抑制過度預測偏好
    """
    current_logits = logits.copy().astype(np.float64)
    for _ in range(max_iter):
        probs = torch.softmax(torch.from_numpy(current_logits), dim=1).numpy()
        pred_dist = probs.mean(axis=0)
        adjustment = np.log(pred_dist + eps)
        current_logits = current_logits - adjustment
    return np.argmax(current_logits, axis=1)


# 專案八新加入：輕量化邊緣端學生模型 (MobileNet 核心)
class DepthwiseSeparableConv(nn.Module):
    """深度可分離卷積：大幅降低晶圓端運算成本"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class WaferStudentMobileNet(nn.Module):
    """機台端即時推論專用輕量模型 (Student Model)"""

    def __init__(self, num_classes=4):
        super().__init__()
        # 初始標準卷積 (對齊 3 通道輸入)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 堆疊深度可分離卷積層 (Depthwise + Pointwise)
        self.dw_layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.dw_layers(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        return self.fc(x)
