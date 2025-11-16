"""
金融異常檢測 - 主流程

本模組實現了使用圖卷積網路和變分自編碼器 (GCN-VAE) 檢測可疑金融帳戶的主訓練流程。該解決方案透過引入時間視窗特徵來捕捉帳戶標記前的行為變化，在私有排行榜上取得了 0.34 的 F1 分數。

架構概述：
- 特徵工程：27 維特徵空間
* 基本特徵（10 維）：度、金額、自交易
* 時間特徵（5 維）：交易時間模式
* 模式特徵（5 維）：合作夥伴多樣性、金額統計
* 時間窗口特徵（6 維）：早期與晚期行為變化
* PageRank（1 維）：網路中心性
- 模型：採用 4 層編碼器的 GCN-VAE
- 訓練：使用 Focal Loss 進行困難負樣本挖掘

主要創新：
1. 時間窗口分析：比較早期（第 1-60 天）和晚期（第 91-121 天）
行為，以檢測預先標記的異常
2. 困難負樣本挖掘：選擇與正樣本相似的具有挑戰性的負樣本，以提高模型穩健性
3. 多目標損失：平衡重構、KL 散度和
分類，並使用 Focal Loss 解決類別不平衡問題
"""
# ============================================================================
# 第1次方案：時序窗口增強版
# 基於：F1-0.277.py
# 新增：6維時序窗口特徵
# 目標：F1 0.30-0.35
# 
# 核心改進：
# 1. 時序窗口分析（早期 vs 晚期行為）
# 2. 變化率特徵（抓住「即將被標記前」的異常變化）
# 3. 最後交易日特徵（越接近day 121越可疑）
# ============================================================================

from numba import njit, prange
import time

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

print("✓ 套件載入完成")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# ============================================================================
# Cell 2: 參數設定（保持F1-0.277的成功參數）
# ============================================================================

TRAIN_CSV = './dataset/acct_transaction.csv'
ALERT_CSV = './dataset/acct_alert.csv'
PREDICT_CSV = './dataset/acct_predict.csv'
OUTPUT_CSV = './output/predictions_plan1_temporal.csv'
CHECKPOINT_DIR = './checkpoints_plan1'

# === 模型參數（F1-0.277的成功配置）===
HIDDEN_DIMS = [128, 64, 32, 16]  # GCN layer dimensions
DROPOUT = 0.5  # Dropout rate for regularization
EPOCHS = 150  # Maximum number of training epochs
LEARNING_RATE = 0.01  # Adam optimizer learning rate
WEIGHT_DECAY = 5e-4  # L2 regularization weight
PATIENCE = 30  # Early stopping patience (critical parameter for F1 0.34)

# === 損失權重 ===
KL_WEIGHT = 0.3  # Weight for KL divergence loss in VAE
CLS_WEIGHT = 5.0  # Weight for classification loss (handles class imbalance)
FOCAL_ALPHA = 0.75  # Focal loss alpha (focus on hard examples)
FOCAL_GAMMA = 2.0  # Focal loss gamma (down-weight easy examples)

# === 資料參數 ===
NEG_RATIO = 5.0  # Negative to positive sample ratio
EDGE_SAMPLES = 8000  # Number of graph edges to sample per training epoch
MAX_EDGES = None  # Maximum edges to load (None = load all)

# === 其他 ===
USE_ENSEMBLE = False  # Whether to use model ensemble (not implemented)
RANDOM_SEED = 42  # Random seed for reproducibility
DEVICE = 'cpu'  # Computing device ('cpu')

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print("="*70)
print("  🎯 第1次方案：時序窗口增強版")
print("="*70)
print(f"基於：F1-0.277.py 成功配置")
print(f"新增：6維時序窗口特徵")
print(f"總特徵：21 + 6 = 27維")
print(f"")
print(f"時序窗口設計：")
print(f"  - 早期：day 1-60")
print(f"  - 中期：day 61-90")
print(f"  - 晚期：day 91-121 ⭐ 關鍵窗口")
print(f"")
print(f"目標：F1 0.30-0.35")
print("="*70)

# ============================================================================
# 執行主程序
# ============================================================================

from Preprocess.data_preprocess import (
    load_transaction_data,
    build_account_mapping,
    build_all_features
)

from Model.model import (
    build_labels_with_hard_negatives,
    sparse_to_torch,
    train_model,
    predict
)

print("\n" + "="*70)
print("  階段 1: 數據載入")
print("="*70)

df_txn = load_transaction_data(TRAIN_CSV, max_edges=MAX_EDGES)
id2idx, idx2id = build_account_mapping(df_txn)

print("\n" + "="*70)
print("  階段 2: 特徵工程")
print("="*70)

features, adj, adj_norm = build_all_features(df_txn, id2idx)

print(f"\n最終數據摘要:")
print(f"  - 節點數: {len(id2idx):,}")
print(f"  - 邊數: {adj.nnz:,}")
print(f"  - 特徵維度: {features.shape}")

print("\n" + "="*70)
print("  階段 3: 建立標籤")
print("="*70)

train_idx, train_y, val_idx, val_y = build_labels_with_hard_negatives(
    ALERT_CSV, id2idx, features, neg_ratio=NEG_RATIO, seed=RANDOM_SEED, val_ratio=0.2
)

print("\n" + "="*70)
print("  階段 4: 模型訓練")
print("="*70)

config = {
    'device': DEVICE, 'hidden_dims': HIDDEN_DIMS, 'dropout': DROPOUT,
    'epochs': EPOCHS, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
    'patience': PATIENCE, 'kl_weight': KL_WEIGHT, 'cls_weight': CLS_WEIGHT,
    'focal_alpha': FOCAL_ALPHA, 'focal_gamma': FOCAL_GAMMA, 'edge_samples': EDGE_SAMPLES
}

model, clf, best_threshold, best_f1 = train_model(
    features, adj, adj_norm, train_idx, train_y, val_idx, val_y, config, seed=RANDOM_SEED
)

print("\n✓ 訓練完成！")

print("\n" + "="*70)
print("  階段 5: 驗證集評估")
print("="*70)

device = torch.device(DEVICE)
features_t = torch.tensor(features, dtype=torch.float32, device=device)
adj_norm_t = sparse_to_torch(adj_norm).coalesce().to(device)
val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)

predictions, probs = predict(model, clf, features_t, adj_norm_t, val_idx_t, best_threshold)

val_y_np = val_y_t.cpu().numpy()
p, r, f1, _ = precision_recall_fscore_support(val_y_np, predictions, average='binary', zero_division=0)

print(f"\n最終驗證集結果 (閾值={best_threshold:.2f}):")
print(f"  Precision: {p:.4f}")
print(f"  Recall: {r:.4f}")
print(f"  F1 Score: {f1:.4f}")

print("\n" + "="*70)
print("  階段 6: 預測")
print("="*70)

df_pred = pd.read_csv(PREDICT_CSV)
accts = df_pred['acct'].astype(str).tolist()
print(f"  ✓ 載入 {len(accts):,} 個待預測帳戶")

pred_indices = []
for acct in accts:
    if acct in id2idx:
        pred_indices.append(id2idx[acct])
    else:
        pred_indices.append(-1)

valid_mask = np.array(pred_indices) >= 0
valid_indices = np.array([i for i in pred_indices if i >= 0])

valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=device)
predictions_valid, probs_valid = predict(model, clf, features_t, adj_norm_t, valid_indices_t, best_threshold)

predictions = np.zeros(len(accts), dtype=int)
probs_all = np.zeros(len(accts), dtype=float)

valid_idx = 0
for i, is_valid in enumerate(valid_mask):
    if is_valid:
        predictions[i] = predictions_valid[valid_idx]
        probs_all[i] = probs_valid[valid_idx]
        valid_idx += 1

result_df = pd.DataFrame({'acct': accts, 'label': predictions})
result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ 預測結果已儲存至: {OUTPUT_CSV}")
print(f"\n預測統計:")
print(f"  預測為異常: {predictions.sum():,} ({predictions.sum()/len(accts)*100:.2f}%)")

print("\n" + "="*70)
print("  🎯 第1次方案執行完成！")
print("="*70)
print(f"驗證集F1: {f1:.4f}")
print(f"最佳閾值: {best_threshold:.2f}")
print(f"輸出檔案: {OUTPUT_CSV}")
print(f"")
print(f"⭐ 新增時序特徵：")
print(f"  1. 早期交易頻率")
print(f"  2. 晚期交易頻率")
print(f"  3. 頻率變化率 (抓住突然活躍)")
print(f"  4. 晚期平均金額")
print(f"  5. 金額變化率 (抓住突然大額)")
print(f"  6. 最後交易日 (越接近day 121越可疑)")

print("="*70)
