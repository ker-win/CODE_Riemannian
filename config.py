# -*- coding: utf-8 -*-
"""
黎曼空間 MI-BCI 分類方法 - 配置檔案

基於論文：Study of MI-BCI classification method based on the 
Riemannian transform of personalized EEG spatiotemporal features
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# === 路徑配置 ===
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Dataset"
RESULTS_DIR = BASE_DIR / "results_riemannian"

# === 時間窗口配置 T1-T8 ===
@dataclass
class TimeWindowConfig:
    """多時間窗口配置 (窗長1s, 步長0.5s)"""
    # T1-T6: 滑動窗口
    # T7: MI 任務期間
    # T8: 完整期間
    windows: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1.0),   # T1
        (0.5, 1.5),   # T2
        (1.0, 2.0),   # T3
        (1.5, 2.5),   # T4
        (2.0, 3.0),   # T5
        (2.5, 3.5),   # T6
        (0.5, 2.5),   # T7 - MI 任務期間
        (0.0, 4.0),   # T8 - 完整期間
    ])
    
DEFAULT_TIME_WINDOWS = TimeWindowConfig()

# === 頻帶配置 B1-B64 ===
def generate_filter_bands() -> List[Tuple[float, float]]:
    """
    生成 64 個頻帶
    頻率範圍: 4-40 Hz
    頻帶寬度: 2, 4, 8, 16, 32 Hz
    滑動步長: 2 Hz
    """
    bands = []
    for bandwidth in [2, 4, 8, 16, 32]:
        for start in range(4, 40, 2):
            if start + bandwidth <= 40:
                bands.append((float(start), float(start + bandwidth)))
    return bands

FILTER_BANDS = generate_filter_bands()  # 64 個頻帶

# === 黎曼特徵配置 ===
@dataclass
class RiemannianConfig:
    """黎曼特徵配置"""
    # Stage A: 粗篩 top-K (T,B) 組合
    top_k_combinations: int = 50  # 從 512 個 (T,B) 組合中選 top-K
    
    # 特徵選擇
    nca_n_components: int = 30  # NCA 降維後的維度
    
DEFAULT_RIEMANNIAN = RiemannianConfig()

# === 分類配置 ===
@dataclass
class ClassifierConfig:
    """分類器配置"""
    cv_folds: int = 5
    random_state: int = 42
    # SVM 參數
    svm_kernel: str = 'linear'  # 論文使用線性 kernel
    svm_C_range: List[float] = field(default_factory=lambda: [0.01, 0.1, 1, 10, 100])

DEFAULT_CLASSIFIER = ClassifierConfig()

# === 預處理配置 ===
@dataclass
class PreprocessConfig:
    """預處理配置"""
    # 濾波器設定
    filter_order: int = 5  # Butterworth 階數
    # Epoch 時間範圍 (用於載入資料)
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0

DEFAULT_PREPROCESS = PreprocessConfig()

# === 快取配置 ===
@dataclass
class CacheConfig:
    """快取配置"""
    enable_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: BASE_DIR / ".cache_riemannian")

DEFAULT_CACHE = CacheConfig()
