"""GCN-VAE 模型架構、訓練和預測模組
該模組實現了用於金融異常檢測的核心機器學習組件，包括：
- 基於變分自編碼器的圖卷積網路 (GCN-VAE)
- 用於穩健訓練的難負樣本挖掘
- 用於處理類別不平衡的 Focal 損失函數
- 結合重構、KL 散度和分類的多目標損失函數
- 用於二元分類的閾值優化

訓練策略：
1. 挖掘難負樣本（50% 難負樣本，50% 隨機負樣本）
2. 提前停止，耐心值 = 30
3. 在驗證集上進行閾值最佳化
4. 梯度裁切（最大歸一化值 = 1.0）
5. 邊緣採樣（每輪 8000 條邊緣）
"""

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

# ============================================================================
# Cell 8-16: 模型訓練（保持F1-0.277的成功配置）
# ============================================================================

def build_labels_with_hard_negatives(alert_csv, id2idx, features, 
                                     neg_ratio=5.0, seed=42, val_ratio=0.2):
    """Hard Negative Mining建立標籤"""
    print("\n" + "="*70)
    print("  建立訓練標籤 (Hard Negative Mining)")
    print("="*70)
    
    rng = np.random.default_rng(seed)
    df = pd.read_csv(alert_csv)
    
    pos_ids = [id2idx[str(a)] for a in df['acct'].values if str(a) in id2idx]
    
    if len(pos_ids) == 0:
        raise ValueError("沒有找到任何正樣本帳戶")
    
    print(f"正樣本數: {len(pos_ids):,}")
    
    all_nodes = np.arange(len(id2idx))
    pos_set = set(pos_ids)
    neg_pool = np.array([i for i in all_nodes if i not in pos_set])
    
    # Hard Negative Mining
    pos_features = features[pos_ids]
    pos_mean = pos_features.mean(axis=0)
    
    neg_features = features[neg_pool]
    similarities = np.dot(neg_features, pos_mean)
    
    n_neg = int(len(pos_ids) * neg_ratio)
    n_neg = min(n_neg, len(neg_pool))
    
    n_hard = n_neg // 2
    n_random = n_neg - n_hard
    
    hard_candidate_indices = np.argsort(similarities)[-n_hard*3:]
    hard_neg_indices = rng.choice(hard_candidate_indices, size=n_hard, replace=False)
    hard_neg = neg_pool[hard_neg_indices]
    
    remaining_pool = np.setdiff1d(neg_pool, hard_neg)
    random_neg = rng.choice(remaining_pool, size=n_random, replace=False)
    
    neg_ids = np.concatenate([hard_neg, random_neg])
    
    print(f"負樣本數: {n_neg:,} (困難: {n_hard:,}, 隨機: {n_random:,})")
    
    X_idx = np.concatenate([pos_ids, neg_ids])
    y_lab = np.concatenate([
        np.ones(len(pos_ids), dtype=np.float32),
        np.zeros(len(neg_ids), dtype=np.float32)
    ])
    
    perm = rng.permutation(len(X_idx))
    X_idx, y_lab = X_idx[perm], y_lab[perm]
    
    n_val = int(len(X_idx) * val_ratio)
    train_idx = X_idx[n_val:]
    train_y = y_lab[n_val:]
    val_idx = X_idx[:n_val]
    val_y = y_lab[:n_val]
    
    print(f"\n訓練集: {len(train_idx):,} (正樣本: {train_y.sum():.0f})")
    print(f"驗證集: {len(val_idx):,} (正樣本: {val_y.sum():.0f})")
    
    return train_idx, train_y, val_idx, val_y


class GCNLayer(nn.Module):
    """Single Graph Convolutional Network (GCN) layer 單圖卷積網（GCN）層"""
    def __init__(self, in_features, out_features, dropout=0.0, use_activation=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = dropout
        self.use_activation = use_activation
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.use_activation:
            output = F.relu(output)
        return output


class GCNVAE(nn.Module):
    """Graph Convolutional Variational Autoencoder. 圖卷積變分自編碼器"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout=0.5):
        super(GCNVAE, self).__init__()
        self.gc1 = GCNLayer(input_dim, hidden_dims[0], dropout, True)
        self.gc2 = GCNLayer(hidden_dims[0], hidden_dims[1], dropout, True)
        self.gc3 = GCNLayer(hidden_dims[1], hidden_dims[2], dropout, True)
        self.gc_mu = GCNLayer(hidden_dims[2], hidden_dims[3], dropout, False)
        self.gc_logvar = GCNLayer(hidden_dims[2], hidden_dims[3], dropout, False)
    
    def encode(self, x, adj):
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        h3 = self.gc3(h2, adj)
        mu = self.gc_mu(h3, adj)
        logvar = self.gc_logvar(h3, adj)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """針對類別不平衡的焦點損失函數：FL = -α(1-p_t)^γ log(p_t)"""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
    loss = alpha_weight * focal_weight * bce_loss
    return loss.mean()


def reconstruction_loss(z, pos_edges, neg_edges):
    """透過鏈路預測進行圖重建損失"""
    pos_logits = (z[pos_edges[:, 0]] * z[pos_edges[:, 1]]).sum(dim=1)
    neg_logits = (z[neg_edges[:, 0]] * z[neg_edges[:, 1]]).sum(dim=1)
    
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    
    return pos_loss + neg_loss


def kl_divergence(mu, logvar):
    """KL 散度: KL[q(z|x) || N(0,I)] = -0.5 * Σ(1 + logvar - μ² - exp(logvar))"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def sample_edges(adj, num_samples, rng):
    """為圖的重構選取正負邊樣本"""
    A = sp.triu(adj, 1).tocoo()
    pos_edges = np.vstack([A.row, A.col]).T
    
    if len(pos_edges) == 0:
        raise ValueError("沒有正邊可以採樣")
    
    num = min(num_samples, len(pos_edges))
    pos_sel = pos_edges[rng.choice(len(pos_edges), num, replace=False)]
    
    existing = set(map(tuple, pos_edges))
    n = adj.shape[0]
    neg = []
    attempts = 0
    max_attempts = num * 10
    
    while len(neg) < num and attempts < max_attempts:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in existing:
                neg.append((a, b))
                existing.add((a, b))
        attempts += 1
    
    neg_sel = np.array(neg[:num], dtype=np.int64)
    return pos_sel, neg_sel


@torch.no_grad()
def evaluate_model(model, clf, features, adj_norm, idx, y_true, threshold=0.5):
    """使用綜合指標對給定的指標評估模型"""
    model.eval()
    clf.eval()
    
    mu, _ = model.encode(features, adj_norm)
    logits = clf(mu[idx]).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    y_pred = (probs >= threshold).astype(int)
    
    p, r, f1, _ = precision_recall_fscore_support(y_true_np, y_pred, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_np, probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_true_np, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'auc': float(auc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn), 'probs': probs}


def find_best_threshold(model, clf, features, adj_norm, idx, y_true):
    """透過網格搜尋找到最佳分類閾值"""
    model.eval()
    clf.eval()
    
    with torch.no_grad():
        mu, _ = model.encode(features, adj_norm)
        logits = clf(mu[idx]).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    y_true_np = y_true.cpu().numpy()
    
    best_f1 = 0
    best_thr = 0.5
    
    for thr in np.arange(0.1, 0.9, 0.05):
        y_pred = (probs >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true_np, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    return best_thr, best_f1

def sparse_to_torch(sparse_mx):
    """將 scipy 稀疏矩陣轉換為 PyTorch 稀疏張量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def train_model(features, adj, adj_norm, train_idx, train_y, val_idx, val_y, config, seed=42):
    """使用多目標損失函數和早停法訓練 GCN-VAE 模型。
    達成完整的訓練流程：
    1. 初始化模型與優化器
    2. 使用重建損失函數 + KL 損失函數 + 分類損失函數進行訓練
    3. 每 5 個 epoch 在驗證集上進行評估
    4. 耐心早停
    5. 尋找最優分類閾值
    """
    print(f"\n{'='*70}")
    print(f"  訓練模型 (seed={seed})")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device(config['device'])
    
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    adj_norm_t = sparse_to_torch(adj_norm).coalesce().to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
    
    model = GCNVAE(input_dim=features.shape[1], hidden_dims=config['hidden_dims'], dropout=config['dropout']).to(device)
    clf = nn.Linear(config['hidden_dims'][-1], 1).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(clf.parameters()), 
                          lr=config['lr'], weight_decay=config['weight_decay'])
    
    rng = np.random.default_rng(seed)
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    best_clf_state = None
    
    print(f"\n開始訓練...")
    pbar = trange(1, config['epochs'] + 1, desc="Training")
    
    for epoch in pbar:
        model.train()
        clf.train()
        
        optimizer.zero_grad()
        
        z, mu, logvar = model(features_t, adj_norm_t)
        
        pos_e, neg_e = sample_edges(adj, config['edge_samples'], rng)
        pos_e_t = torch.tensor(pos_e, dtype=torch.long, device=device)
        neg_e_t = torch.tensor(neg_e, dtype=torch.long, device=device)
        loss_rec = reconstruction_loss(z, pos_e_t, neg_e_t)
        
        logits_train = clf(z[train_idx_t]).squeeze(1)
        loss_cls = focal_loss(logits_train, train_y_t, config['focal_alpha'], config['focal_gamma'])
        
        loss_kl = kl_divergence(mu, logvar)
        
        total_loss = loss_rec + config['kl_weight'] * loss_kl + config['cls_weight'] * loss_cls
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_model(model, clf, features_t, adj_norm_t, val_idx_t, val_y_t, threshold=0.5)
            
            pbar.set_postfix({'loss': f"{total_loss.item():.3f}", 'val_F1': f"{metrics['f1']:.3f}",
                            'val_P': f"{metrics['precision']:.3f}", 'val_R': f"{metrics['recall']:.3f}"})
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict()
                best_clf_state = clf.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience'] // 5:
                print(f"\n  ✓ Early stopping at epoch {epoch}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        clf.load_state_dict(best_clf_state)
    
    print(f"\n  尋找最佳閾值...")
    best_thr, best_thr_f1 = find_best_threshold(model, clf, features_t, adj_norm_t, val_idx_t, val_y_t)
    
    print(f"  ✓ 最佳驗證F1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"  ✓ 最佳閾值={best_thr:.2f}, 使用該閾值的F1={best_thr_f1:.4f}")
    
    return model, clf, best_thr, best_thr_f1


@torch.no_grad()
def predict(model, clf, features, adj_norm, idx, threshold=0.5):
    """根據給定的節點索引進行預測"""
    model.eval()
    clf.eval()
    
    mu, _ = model.encode(features, adj_norm)
    logits = clf(mu[idx]).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    
    predictions = (probs >= threshold).astype(int)
    
    return predictions, probs
