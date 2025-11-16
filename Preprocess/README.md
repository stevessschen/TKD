# data_preprocess.py - 特徵工程模組（27 維特徵提取）

12個函數，包括：

load_transaction_data() - 數據載入

build_account_mapping() - 帳戶映射

extract_basic_features() - 基礎特徵 (10維)

extract_time_features() - 時間特徵 (5維)

extract_pattern_features() - 模式特徵 (5維)

extract_temporal_window_features_fast() - 時序窗口特徵 (6維) ⭐ 關鍵創新

_compute_temporal_stats_numba() - Numba加速核心

build_graph() - 圖結構構建

compute_pagerank() - PageRank計算

normalize_adj() - GCN正規化

build_all_features() - 完整流程整

## 突顯競賽關鍵創新
### 特別強調：

⭐ 時序窗口分析 (早期 vs 晚期行為) - F1 從 0.277 → 0.34

⭐ Hard Negative Mining - 改善模型泛化能力

⭐ Focal Loss - 處理嚴重類別不平衡

⭐ Numba加速 - 30分鐘 → 2分鐘 (15-20x speedup)
