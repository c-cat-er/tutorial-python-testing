import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from tqdm import tqdm

# 引入上面剛寫好的模組與資料集
from wafer_map.dl_models import (
    ConvAutoEncoder,
    DomainClassifier,
    FeatureExtractor,
    LabelPredictor,
    PseudoDataset,
    WaferAnomalyDataset,
    WaferImageDataset,
    WaferResNetClassifier,
    WaferVAE,
    calibrate_iterative,
    get_wafer_transforms,
    loss_vae,
)


# 複用筆記核心：高信心偽標籤篩選機制 (v2 修正版：回傳索引+標籤)
def get_pseudo_labels_v2(dataset, model, threshold=0.85, batch_size=32, device="cpu"):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    softmax = nn.Softmax(dim=-1)
    pseudo_indices_list, pseudo_labels_list = [], []
    sample_idx = 0

    for batch in data_loader:
        imgs, _ = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
            probs = softmax(logits)
            max_probs, pred_labels = torch.max(probs, dim=1)
            mask = max_probs > threshold
            if mask.any():
                batch_size_actual = len(imgs)
                batch_indices = torch.arange(
                    sample_idx, sample_idx + batch_size_actual, device="cpu"
                )[mask.cpu()]
                pseudo_indices_list.append(batch_indices)
                pseudo_labels_list.append(
                    pred_labels[mask].detach().cpu()
                )  # 斷開梯度，防止記憶體洩漏
            sample_idx += len(imgs)

    model.train()  # 還原訓練狀態
    if len(pseudo_indices_list) == 0:
        return [], torch.tensor([])
    return torch.cat(pseudo_indices_list, dim=0).tolist(), torch.cat(
        pseudo_labels_list, dim=0
    )


def classify_defects(wafer_matrices: list, labels: list = None, config: dict = None):
    """
    專案三升級版：晶圓缺陷 ResNet18 分類器 (含半監督機制)
    wafer_matrices: 由 loader.wafer_to_matrix 產出的 2D 矩陣列表
    labels: 整數型標籤列表 (0:Ring, 1:Scratch, 2:Cluster, 3:Random)，無標籤時傳入 None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = config.get("num_classes", 4) if config else 4
    batch_size = config.get("batch_size", 32) if config else 32
    epochs = config.get("epochs", 40) if config else 40
    threshold = config.get("threshold", 0.85) if config else 0.85

    train_tfm, test_tfm = get_wafer_transforms()
    model = WaferResNetClassifier(num_classes=num_classes).to(device)

    # 產線情境 A：純推論模式 (線上即時預測)
    if labels is None:
        model.eval()
        infer_dataset = WaferImageDataset(wafer_matrices, transform=test_tfm)
        infer_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in infer_loader:
                imgs, _ = batch
                logits = model(imgs.to(device))
                predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        return predictions, 1.0

    # 產線情境 B：模型重訓模式 (含半監督雙視圖訓練)
    # 將資料切分為有標籤訓練集與獨立驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        wafer_matrices, labels, test_size=0.2, random_state=42
    )

    # 建立弱視圖與強視圖資料集
    labeled_train_set = WaferImageDataset(X_train, y_train, transform=train_tfm)
    val_set = WaferImageDataset(X_val, y_val, transform=test_tfm)

    # 假設其餘未被選入訓練的矩陣作為無標籤池 (模擬 unlabeled_set)
    unlabeled_set_train = WaferImageDataset(wafer_matrices, transform=train_tfm)
    unlabeled_set_pseudo = WaferImageDataset(wafer_matrices, transform=test_tfm)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )  # 沿用筆記：標籤平滑化防止過度自信
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        current_train_set = labeled_train_set

        # 5個 Epoch 熱身結束後，動態重組資料集
        if epoch >= 5:
            pseudo_indices, pseudo_labels = get_pseudo_labels_v2(
                unlabeled_set_pseudo,
                model,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
            )
            if len(pseudo_indices) > 0:
                strong_pseudo_images_set = Subset(unlabeled_set_train, pseudo_indices)
                pseudo_set = PseudoDataset(strong_pseudo_images_set, pseudo_labels)
                current_train_set = ConcatDataset(
                    [labeled_train_set, pseudo_set]
                )  # 拼接偽標籤

        train_loader = DataLoader(
            current_train_set, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Train Loop
        model.train()
        for batch in train_loader:
            imgs, labels_batch = batch
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation Loop
        model.eval()
        val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels_batch = batch
                logits = model(imgs.to(device))
                acc = (logits.argmax(dim=-1).cpu() == labels_batch).float().mean()
                val_accs.append(acc.item())

        current_val_acc = np.mean(val_accs)
        if current_val_acc > best_acc:
            best_acc = current_val_acc
            # 可在此處持久化保存最佳權重 yield_resnet_best.pth

    return model, float(best_acc)


def detect_unknown_defects(wafer_matrices: list, config: dict):
    """
    專案六：無監督未知缺陷識別控制台
    wafer_matrices: 2D 晶圓矩陣列表
    config: settings.yaml 讀入的異常檢測配置參數
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = config.get("model_type", "cnn")
    img_size = config.get("img_size", 64)
    save_path = "data/merged/models/anomaly_detector.pt"  # 對齊 Page 2 的 data/merged/models 規範！

    model_classes = {"cnn": ConvAutoEncoder(), "vae": WaferVAE()}
    model = model_classes[model_type].to(device)

    # 建立 DataLoader
    dataset = WaferAnomalyDataset(wafer_matrices, img_size=img_size)
    loader = DataLoader(
        dataset, batch_size=config.get("batch_size", 128), shuffle=False
    )

    # 載入現有模型權重進行無監督推論 (實際產線佈署多為此情境)
    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
    except FileNotFoundError:
        # 若無權重則現場快速做無監督自我熱身訓練 (Cold-start Adaptive Training)
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(10):  # 快速迭代
            for img in loader:
                img = img.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = (
                    loss_vae(output[0], img, output[1], output[2])
                    if model_type == "vae"
                    else criterion(output, img)
                )
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), save_path)

    model.eval()
    eval_loss = torch.nn.MSELoss(reduction="none")  # 關鍵：計算單片 Pixel-wise 獨立誤差
    anomaly_scores = []

    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            output = model(img)
            if model_type == "vae":
                output = output[0]

            # 數學核心：計算原圖與重建圖在每個像素高度、寬度與通道上的平方差總和 (sum([1,2,3]))
            loss = eval_loss(output, img).flatten(start_dim=1).sum(dim=1)
            anomaly_scores.extend(
                torch.sqrt(loss).cpu().numpy().tolist()
            )  # 開根號轉為標準 RMSE

    results = []
    threshold = config.get("anomaly_threshold", 15.5)
    for score in anomaly_scores:
        results.append(
            {
                "anomaly_score": score,
                "is_unknown_defect": score > threshold,  # 超過產線安全門檻則回傳 True
            }
        )
    return results


# 專案七
def train_domain_adaptation(
    factory_a_matrices: list,
    factory_a_labels: list,
    factory_b_matrices: list,
    epochs=20,
    lamb=0.1,
):
    """
    專案七：跨廠區模型遷移對抗訓練控制台
    factory_a_matrices: 擁有完整標籤的 A 廠 (Source) 晶圓圖矩陣列表
    factory_a_labels: A 廠缺陷模式標籤 (0~3)
    factory_b_matrices: 毫無標籤的 B 新廠 (Target) 晶圓圖矩陣列表
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 領域特徵前處理管線 (Canny 邊緣對齊與單通道灰階對齊)
    def preprocess_domain(matrices, use_canny=False):
        processed = []
        for m in matrices:
            img = (m * (255.0 / (m.max() if m.max() > 0 else 1.0))).astype(np.uint8)
            img_resized = cv2.resize(img, (32, 32))  # 強制對齊 32x32 分辨率
            if use_canny:
                # 核心複用：利用 Canny 強行濾除 A 廠複雜背景，轉換成與手繪一樣的乾淨形狀輪廓
                img_resized = cv2.Canny(img_resized, 170, 300)
            processed.append(img_resized)
        tensor_data = torch.FloatTensor(np.array(processed)).unsqueeze(
            1
        )  # 轉為 (B, 1, 32, 32)
        return 2.0 * (tensor_data / 255.0) - 1.0  # 歸一化 [-1, 1]

    # A 廠採用 Canny 以抹除機台環境噪訊；B 廠直接取形狀
    src_tensor = preprocess_domain(factory_a_matrices, use_canny=True)
    tgt_tensor = preprocess_domain(factory_b_matrices, use_canny=False)

    # 建立雙域並行數據流 loader
    src_loader = DataLoader(
        TensorDataset(src_tensor, torch.LongTensor(factory_a_labels)),
        batch_size=32,
        shuffle=True,
    )
    tgt_loader = DataLoader(TensorDataset(tgt_tensor), batch_size=32, shuffle=True)

    # 實體化三頭組件
    FE = FeatureExtractor().to(device)
    LP = LabelPredictor(num_classes=4).to(device)
    DC = DomainClassifier().to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    opt_F = optim.Adam(FE.parameters(), lr=1e-4)
    opt_C = optim.Adam(LP.parameters(), lr=1e-4)
    opt_D = optim.Adam(DC.parameters(), lr=1e-4)

    # 開始對抗訓練主迴圈
    for epoch in range(epochs):
        FE.train()
        LP.train()
        DC.train()

        # 完美複用筆記的並行 zip 對齊與雙域對抗優化
        for i, ((src_data, src_label), (tgt_data,)) in enumerate(
            zip(src_loader, tgt_loader)
        ):
            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data = tgt_data.to(device)

            # 核心複用：將兩廠資料在 Batch 進行拼接，維持 BatchNorm 內部動態均值穩定
            mixed_data = torch.cat([src_data, tgt_data], dim=0)
            domain_label = torch.zeros(mixed_data.size(0), 1).to(device)
            domain_label[: src_data.size(0)] = 1.0  # A廠設為 1, B廠設為 0

            # Step 1 : 訓練 Domain Classifier (調用 .detach() 鎖定 Extractor)
            opt_D.zero_grad()
            feature_detached = FE(mixed_data).detach()  # 斷開反向傳播梯度
            domain_logits = DC(feature_detached)
            loss_D = domain_criterion(domain_logits, domain_label)
            loss_D.backward()
            opt_D.step()

            # Step 2 : 訓練 Feature Extractor & Label Classifier (對抗體現)
            opt_F.zero_grad()
            opt_C.zero_grad()

            feature = FE(mixed_data)
            class_logits = LP(feature[: src_data.shape[0]])
            domain_logits_adv = DC(feature)

            # 減號 "-" 類似 GAN 的 Generator 概念，最大化工廠領域分類器的錯誤率，迫使學到通用特徵
            loss_F = class_criterion(class_logits, src_label) - lamb * domain_criterion(
                domain_logits_adv, domain_label
            )
            loss_F.backward()
            opt_F.step()
            opt_C.step()

    # 跨廠區線上即時推論階段 (B 廠盲測推理)
    FE.eval()
    LP.eval()
    all_logits = []
    with torch.no_grad():
        for (tgt_data,) in tgt_loader:
            logits = LP(FE(tgt_data.to(device)))
            all_logits.append(logits.cpu().numpy())

    # 核心複用筆記：利用 Iterative Calibration 對 B 新廠輸出進行分佈校正修正
    all_logits = np.concatenate(all_logits, axis=0)
    calibrated_preds = calibrate_iterative(all_logits, max_iter=5)

    return calibrated_preds.tolist()


import torch.nn.functional as F
from torch.utils.data import DataLoader
from wafer_map.dl_models import (
    WaferImageDataset,
    WaferStudentMobileNet,
    get_wafer_transforms,
)


def distill_teacher_to_student(
    wafer_matrices: list, labels: list, teacher_model, config: dict
):
    """
    專案八：知識蒸餾控制流水線 (Teacher ➔ Student)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = config.get("distill_alpha", 0.7)  # 軟標籤損失權重
    T = config.get("distill_temperature", 4.0)  # 溫度係數：平滑化機台特徵
    batch_size = config.get("batch_size", 64)
    epochs = config.get("distill_epochs", 15)  # 蒸餾收斂極快，一般 10-15 epoch 即可

    train_tfm, _ = get_wafer_transforms()
    train_loader = DataLoader(
        WaferImageDataset(wafer_matrices, labels, transform=train_tfm),
        batch_size=batch_size,
        shuffle=True,
    )

    # 實體化輕量學生模型
    student_model = WaferStudentMobileNet(num_classes=4).to(device)
    teacher_model.eval()  # 鎖定並凍結專家模型
    student_model.train()

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(
        reduction="batchmean"
    )  # 核心複用：比對師生預測機率分佈

    for epoch in range(epochs):
        for imgs, hard_labels in train_loader:
            imgs, hard_labels = imgs.to(device), hard_labels.to(device)
            optimizer.zero_grad()

            # 同時前向傳播
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            student_logits = student_model(imgs)

            # 傳統硬損失 (Hard Loss)
            loss_hard = criterion_hard(student_logits, hard_labels)

            # 知識蒸餾軟損失 (Soft Loss)
            soft_teacher = F.softmax(teacher_logits / T, dim=-1)
            soft_student = F.log_softmax(student_logits / T, dim=-1)
            loss_soft = criterion_soft(soft_student, soft_teacher) * (T**2)

            # 完美權重融合
            loss = alpha * loss_soft + (1.0 - alpha) * loss_hard
            loss.backward()
            optimizer.step()

    # 核心終點：持久化保存僅有 2MB 大小的輕量模型，準備派發至前段測試設備（Prober）
    torch.save(
        student_model.state_dict(), "data/gold/models/edge_mobilenet_deployed.pth"
    )
    return student_model
