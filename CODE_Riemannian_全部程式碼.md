# CODE_Riemannian 程式碼整合
產生時間: 2025-12-16 16:38:21
共 13 個檔案
---
## 目錄
1. [__init__.py](#__init__-py)
2. [classification\__init__.py](#classification-__init__-py)
3. [classification\nca_svm.py](#classification-nca_svm-py)
4. [config.py](#config-py)
5. [export_all_code.py](#export_all_code-py)
6. [features\__init__.py](#features-__init__-py)
7. [features\riemannian.py](#features-riemannian-py)
8. [main.py](#main-py)
9. [preprocessing\__init__.py](#preprocessing-__init__-py)
10. [preprocessing\filterbank.py](#preprocessing-filterbank-py)
11. [preprocessing\multiwindow.py](#preprocessing-multiwindow-py)
12. [utils\__init__.py](#utils-__init__-py)
13. [utils\cache.py](#utils-cache-py)

---

## __init__.py {#__init__-py}
**路徑**: `__init__.py`

```python
# -*- coding: utf-8 -*-
"""
黎曼空間 MI-BCI 分類方法

基於論文：Study of MI-BCI classification method based on the 
Riemannian transform of personalized EEG spatiotemporal features
"""
```

---

## classification\__init__.py {#classification-__init__-py}
**路徑**: `classification\__init__.py`

```python
# -*- coding: utf-8 -*-
"""
Classification 模組
"""
```

---

## classification\nca_svm.py {#classification-nca_svm-py}
**路徑**: `classification\nca_svm.py`

```python
# -*- coding: utf-8 -*-
"""
NCA 特徵選擇 + SVM 分類

1. NCA (Neighborhood Component Analysis) 降維
2. 線性 SVM (One-vs-Rest) 分類
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class NCAFeatureSelector:
    """
    NCA 特徵選擇器
    
    使用 NCA 學習投影矩陣，並根據特徵重要度選擇 top-K 特徵
    """
    
    def __init__(self, n_components: int = 30, max_iter: int = 500, random_state: int = 42):
        """
        初始化
        
        Parameters
        ----------
        n_components : int
            NCA 降維後的維度
        max_iter : int
            NCA 最大迭代次數
        random_state : int
            隨機種子
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.nca_ = None
        self.scaler_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NCAFeatureSelector':
        """
        訓練 NCA
        
        Parameters
        ----------
        X : np.ndarray
            特徵矩陣 (n_samples, n_features)
        y : np.ndarray
            標籤
            
        Returns
        -------
        self
        """
        # 標準化
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # 處理 NaN/Inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 限制 n_components
        n_components = min(self.n_components, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)
        n_components = max(1, n_components)
        
        self.nca_ = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        
        try:
            self.nca_.fit(X_scaled, y)
        except Exception as e:
            # 如果 NCA 失敗，使用恆等變換
            print(f"NCA 訓練失敗: {e}")
            self.nca_ = None
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        應用 NCA 轉換
        
        Parameters
        ----------
        X : np.ndarray
            特徵矩陣
            
        Returns
        -------
        np.ndarray
            降維後的特徵
        """
        if self.scaler_ is None:
            raise RuntimeError("請先呼叫 fit() 方法")
        
        X_scaled = self.scaler_.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.nca_ is not None:
            return self.nca_.transform(X_scaled)
        else:
            # 如果 NCA 失敗，返回標準化後的資料
            return X_scaled
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """訓練並轉換"""
        return self.fit(X, y).transform(X)


class RiemannianSVMClassifier:
    """
    黎曼特徵 + NCA + SVM 分類器
    
    流程：
    1. 標準化
    2. NCA 降維
    3. 線性 SVM (One-vs-Rest)
    """
    
    def __init__(
        self,
        nca_components: int = 128,
        nca_max_iter: int = 500,
        svm_C: float = 1.0,
        svm_kernel: str = 'linear',
        random_state: int = 42
    ):
        """
        初始化
        
        Parameters
        ----------
        nca_components : int
            NCA 降維維度
        nca_max_iter : int
            NCA 最大迭代次數
        svm_C : float
            SVM 正則化參數
        svm_kernel : str
            SVM kernel 類型
        random_state : int
            隨機種子
        """
        self.nca_components = nca_components
        self.nca_max_iter = nca_max_iter
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.random_state = random_state
        
        self.nca_ = None
        self.clf_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RiemannianSVMClassifier':
        """
        訓練分類器
        
        Parameters
        ----------
        X : np.ndarray
            特徵矩陣 (n_samples, n_features)
        y : np.ndarray
            標籤
            
        Returns
        -------
        self
        """
        # NCA
        self.nca_ = NCAFeatureSelector(
            n_components=self.nca_components,
            max_iter=self.nca_max_iter,
            random_state=self.random_state
        )
        X_nca = self.nca_.fit_transform(X, y)
        
        # SVM
        n_classes = len(np.unique(y))
        if n_classes > 2:
            # 多分類使用 OVR
            base_svm = SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                random_state=self.random_state
            )
            self.clf_ = OneVsRestClassifier(base_svm)
        else:
            # 二分類
            self.clf_ = SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                random_state=self.random_state
            )
        
        self.clf_.fit(X_nca, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Parameters
        ----------
        X : np.ndarray
            特徵矩陣
            
        Returns
        -------
        np.ndarray
            預測標籤
        """
        if self.nca_ is None or self.clf_ is None:
            raise RuntimeError("請先呼叫 fit() 方法")
        
        X_nca = self.nca_.transform(X)
        return self.clf_.predict(X_nca)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """計算準確率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def create_nca_svm_pipeline(
    nca_components: int = 30,
    svm_C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    建立 sklearn Pipeline
    
    Returns
    -------
    Pipeline
        標準化 → NCA → SVM
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('nca', NeighborhoodComponentsAnalysis(
            n_components=nca_components,
            random_state=random_state,
            max_iter=100
        )),
        ('svm', SVC(
            C=svm_C,
            kernel='linear',
            random_state=random_state
        ))
    ])
```

---

## config.py {#config-py}
**路徑**: `config.py`

```python
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
    
    # 協方差估計器: 'scm', 'lwf', 'oas' (shrinkage 更穩定)
    cov_estimator: str = 'oas'
    
    # 特徵選擇
    nca_n_components: int = 128  # NCA 降維後的維度 (原本 30 太低)
    nca_max_iter: int = 500  # NCA 最大迭代次數
    
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
```

---

## export_all_code.py {#export_all_code-py}
**路徑**: `export_all_code.py`

```python
# -*- coding: utf-8 -*-
"""
將 CODE_Riemannian 的所有 Python 檔案整合到一個 Markdown 文件
"""

from pathlib import Path
from datetime import datetime


def collect_py_files_to_md(source_dir: str, output_file: str = None):
    """
    將指定目錄下的所有 .py 檔案整合到一個 Markdown 文件
    
    Parameters
    ----------
    source_dir : str
        來源目錄路徑
    output_file : str, optional
        輸出的 Markdown 檔案路徑
    """
    source_path = Path(source_dir)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = source_path / f"all_code_{timestamp}.md"
    else:
        output_file = Path(output_file)
    
    # 收集所有 .py 檔案
    py_files = sorted(source_path.rglob("*.py"))
    
    print(f"找到 {len(py_files)} 個 Python 檔案")
    
    # 整合到 Markdown
    md_content = []
    md_content.append(f"# {source_path.name} 程式碼整合\n")
    md_content.append(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"共 {len(py_files)} 個檔案\n")
    md_content.append("---\n")
    
    # 目錄
    md_content.append("## 目錄\n")
    for i, py_file in enumerate(py_files, 1):
        rel_path = py_file.relative_to(source_path)
        anchor = str(rel_path).replace("/", "-").replace("\\", "-").replace(".", "-")
        md_content.append(f"{i}. [{rel_path}](#{anchor})\n")
    md_content.append("\n---\n")
    
    # 各檔案內容
    for py_file in py_files:
        rel_path = py_file.relative_to(source_path)
        anchor = str(rel_path).replace("/", "-").replace("\\", "-").replace(".", "-")
        
        md_content.append(f"\n## {rel_path} {{#{anchor}}}\n")
        md_content.append(f"**路徑**: `{rel_path}`\n\n")
        md_content.append("```python\n")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            md_content.append(content)
        except Exception as e:
            md_content.append(f"# 讀取失敗: {e}\n")
        
        if not content.endswith('\n'):
            md_content.append('\n')
        md_content.append("```\n")
        md_content.append("\n---\n")
    
    # 寫入檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(md_content)
    
    print(f"已儲存至: {output_file}")
    return output_file


if __name__ == "__main__":
    # CODE_Riemannian 目錄
    code_dir = Path(__file__).parent
    
    # 輸出到 CODE_Riemannian 目錄
    output = code_dir / "CODE_Riemannian_全部程式碼.md"
    
    collect_py_files_to_md(code_dir, output)
```

---

## features\__init__.py {#features-__init__-py}
**路徑**: `features\__init__.py`

```python
# -*- coding: utf-8 -*-
"""
Features 模組
"""
```

---

## features\riemannian.py {#features-riemannian-py}
**路徑**: `features\riemannian.py`

```python
# -*- coding: utf-8 -*-
"""
黎曼幾何特徵提取

使用 pyriemann 套件計算：
1. 協方差矩陣 (SCM)
2. 黎曼均值
3. 切空間投影 (Tangent Space Mapping)
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

# 嘗試導入 pyriemann
try:
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.tangentspace import TangentSpace
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False
    warnings.warn("pyriemann 未安裝，將使用簡化版本。請執行: pip install pyriemann")


def compute_covariance(X: np.ndarray, estimator: str = 'scm') -> np.ndarray:
    """
    計算協方差矩陣
    
    Parameters
    ----------
    X : np.ndarray
        EEG 資料 (n_trials, n_channels, n_samples)
    estimator : str
        估計方法 ('scm', 'lwf', 'oas', 'corr')
        
    Returns
    -------
    np.ndarray
        協方差矩陣 (n_trials, n_channels, n_channels)
    """
    if PYRIEMANN_AVAILABLE:
        cov = Covariances(estimator=estimator)
        return cov.fit_transform(X)
    else:
        # 簡化版本：使用 numpy 計算 SCM
        n_trials = X.shape[0]
        n_channels = X.shape[1]
        covs = np.zeros((n_trials, n_channels, n_channels))
        
        for i in range(n_trials):
            covs[i] = np.cov(X[i])
            # 確保正定
            covs[i] = _regularize_cov(covs[i])
        
        return covs


def _regularize_cov(cov: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """正則化協方差矩陣確保正定"""
    return cov + reg * np.eye(cov.shape[0])


class RiemannianFeatures:
    """
    黎曼幾何特徵提取器
    
    流程：
    1. 計算協方差矩陣
    2. 計算黎曼均值 (在 fit 時)
    3. 切空間投影
    """
    
    def __init__(self, estimator: str = 'scm'):
        """
        初始化
        
        Parameters
        ----------
        estimator : str
            協方差估計方法
        """
        self.estimator = estimator
        self.reference_ = None  # 黎曼均值 (參考點)
        self._ts = None  # TangentSpace 物件
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RiemannianFeatures':
        """
        計算黎曼均值作為參考點
        
        Parameters
        ----------
        X : np.ndarray
            EEG 資料 (n_trials, n_channels, n_samples)
        y : np.ndarray, optional
            標籤 (未使用)
            
        Returns
        -------
        self
        """
        # 計算協方差矩陣
        covs = compute_covariance(X, self.estimator)
        
        if PYRIEMANN_AVAILABLE:
            # 使用 pyriemann 的 TangentSpace
            self._ts = TangentSpace()
            self._ts.fit(covs)
            self.reference_ = self._ts.reference_
        else:
            # 簡化版本：使用算術平均
            self.reference_ = np.mean(covs, axis=0)
            self.reference_ = _regularize_cov(self.reference_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        將資料投影到切空間
        
        Parameters
        ----------
        X : np.ndarray
            EEG 資料 (n_trials, n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            切空間特徵 (n_trials, n_features)
            n_features = n_channels * (n_channels + 1) / 2
        """
        if self.reference_ is None:
            raise RuntimeError("請先呼叫 fit() 方法")
        
        # 計算協方差矩陣
        covs = compute_covariance(X, self.estimator)
        
        if PYRIEMANN_AVAILABLE and self._ts is not None:
            return self._ts.transform(covs)
        else:
            # 簡化版本：對數映射 + 向量化
            return self._tangent_space_simple(covs)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """訓練並轉換"""
        return self.fit(X, y).transform(X)
    
    def _tangent_space_simple(self, covs: np.ndarray) -> np.ndarray:
        """
        簡化版切空間投影
        
        使用對數映射: S_i = log(C_ref^{-1/2} @ C_i @ C_ref^{-1/2})
        """
        n_trials = covs.shape[0]
        n_channels = covs.shape[1]
        
        # 計算參考點的逆平方根
        eigvals, eigvecs = np.linalg.eigh(self.reference_)
        eigvals = np.maximum(eigvals, 1e-10)
        ref_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        features = []
        for i in range(n_trials):
            # 白化
            whitened = ref_invsqrt @ covs[i] @ ref_invsqrt
            # 對數映射
            eigvals_w, eigvecs_w = np.linalg.eigh(whitened)
            eigvals_w = np.maximum(eigvals_w, 1e-10)
            log_whitened = eigvecs_w @ np.diag(np.log(eigvals_w)) @ eigvecs_w.T
            # 向量化 (上三角)
            vec = self._symmetric_to_vector(log_whitened)
            features.append(vec)
        
        return np.array(features)
    
    @staticmethod
    def _symmetric_to_vector(mat: np.ndarray) -> np.ndarray:
        """將對稱矩陣轉換為向量 (取上三角)"""
        n = mat.shape[0]
        idx = np.triu_indices(n)
        return mat[idx]


def extract_riemannian_features_multi(
    window_band_data: List[List[np.ndarray]],
    estimator: str = 'scm'
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    對所有 (時間窗口, 頻帶) 組合提取黎曼特徵
    
    Parameters
    ----------
    window_band_data : List[List[np.ndarray]]
        window_band_data[w][b] = (n_trials, n_channels, n_samples)
    estimator : str
        協方差估計方法
        
    Returns
    -------
    all_features : np.ndarray
        形狀 (n_trials, n_combinations, n_tangent_features)
    combinations : List[Tuple[int, int]]
        (window_idx, band_idx) 組合列表
    """
    n_windows = len(window_band_data)
    n_bands = len(window_band_data[0]) if n_windows > 0 else 0
    
    all_features = []
    combinations = []
    
    for w_idx in range(n_windows):
        for b_idx in range(n_bands):
            data = window_band_data[w_idx][b_idx]
            
            # 計算協方差矩陣
            covs = compute_covariance(data, estimator)
            
            # 暫時先收集協方差，之後再統一做切空間
            all_features.append(covs)
            combinations.append((w_idx, b_idx))
    
    # all_features 現在是 list of (n_trials, Nc, Nc)
    # 轉換為 (n_trials, n_combinations, Nc, Nc)
    n_trials = all_features[0].shape[0]
    n_combinations = len(all_features)
    n_channels = all_features[0].shape[1]
    
    covs_all = np.zeros((n_trials, n_combinations, n_channels, n_channels))
    for i, cov in enumerate(all_features):
        covs_all[:, i, :, :] = cov
    
    return covs_all, combinations
```

---

## main.py {#main-py}
**路徑**: `main.py`

```python
# -*- coding: utf-8 -*-
"""
黎曼空間 MI-BCI 分類方法 - 主程式

基於論文：Study of MI-BCI classification method based on the 
Riemannian transform of personalized EEG spatiotemporal features

流程：
1. 多時間窗口切分 (T1-T8)
2. 濾波器組 (B1-B64)
3. 黎曼幾何特徵 (協方差 → 切空間)
4. 兩階段特徵選擇 (粗篩 + NCA)
5. SVM 分類
"""

import sys
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 設定路徑 - 重要：CODE_Riemannian 必須在 CODE 之前
CODE_DIR = Path(__file__).parent
BASE_DIR = CODE_DIR.parent

# 先移除可能存在的 CODE 路徑
code_path = str(BASE_DIR / "CODE")
if code_path in sys.path:
    sys.path.remove(code_path)

# 確保 CODE_Riemannian 在最前面
if str(CODE_DIR) in sys.path:
    sys.path.remove(str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR))

# 忽略警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 先導入本地 config (CODE_Riemannian/config.py)
from config import (
    DEFAULT_TIME_WINDOWS, FILTER_BANDS, 
    DEFAULT_RIEMANNIAN, DEFAULT_CLASSIFIER, DEFAULT_PREPROCESS
)
from preprocessing.multiwindow import MultiWindowSegmenter
from preprocessing.filterbank import FilterBank, apply_filterbank_to_windows
from features.riemannian import RiemannianFeatures, compute_covariance
from classification.nca_svm import RiemannianSVMClassifier

# 現在加入 CODE 路徑以複用 datasets
sys.path.append(str(BASE_DIR / "CODE"))
from datasets.factory import get_dataset


def run_riemannian_classification(
    dataset_name: str = 'BCICIV_2a',
    n_folds: int = 5,
    top_k_combinations: int = 50,
    nca_components: int = 30,
    verbose: bool = True
) -> dict:
    """
    執行黎曼空間分類
    
    Parameters
    ----------
    dataset_name : str
        資料集名稱
    n_folds : int
        交叉驗證折數
    top_k_combinations : int
        選取的 (T,B) 組合數量
    nca_components : int
        NCA 降維維度
    verbose : bool
        是否輸出詳細資訊
        
    Returns
    -------
    dict
        包含每位受試者結果
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"資料集: {dataset_name}")
        print(f"方法: 黎曼空間 + NCA + SVM")
        print(f"時間窗口: 8 個 (T1-T8)")
        print(f"頻帶: 64 個 (B1-B64)")
        print(f"{'='*60}")
    
    # 載入資料集
    try:
        dataset = get_dataset(dataset_name)
    except Exception as e:
        print(f"錯誤: 無法載入資料集 {dataset_name}: {e}")
        return {}
    
    results = {}
    
    # 使用進度條顯示受試者處理進度
    subject_pbar = tqdm(dataset.subjects, desc="受試者", unit="人")
    
    for subject in subject_pbar:
        try:
            subject_pbar.set_postfix({"當前": subject})
            
            # 載入資料 (0-4秒)
            if dataset_name == 'BCICIV_2a':
                epoch_data = dataset.load_binary(
                    subject, session='T',
                    tmin=DEFAULT_PREPROCESS.epoch_tmin,
                    tmax=DEFAULT_PREPROCESS.epoch_tmax
                )
            else:
                epoch_data = dataset.load_subject(subject)
                if hasattr(epoch_data, 'filter_classes'):
                    epoch_data = epoch_data.filter_classes([0, 1])
            
            X = epoch_data.data
            y = epoch_data.labels
            sfreq = epoch_data.sfreq
            
            # === 特徵提取 + 5-Fold 交叉驗證 ===
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import accuracy_score
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_accuracies = []
            
            # 交叉驗證進度條
            fold_pbar = tqdm(
                enumerate(cv.split(X, y)), 
                total=n_folds, 
                desc=f"  {subject} CV", 
                leave=False,
                unit="fold"
            )
            
            for fold_idx, (train_idx, test_idx) in fold_pbar:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # ============================================
                # 修正：先濾波再切窗 (避免短窗低頻濾波邊界效應)
                # ============================================
                from preprocessing.filterbank import FilterBank
                from pyriemann.estimation import Covariances
                from pyriemann.tangentspace import TangentSpace
                
                fb = FilterBank(bands=FILTER_BANDS)
                
                # 1. 先對完整資料做濾波器組
                train_bands = fb.transform(X_train, sfreq)  # List[np.ndarray], 64 個頻帶
                test_bands = fb.transform(X_test, sfreq)
                
                # 2. 對每個頻帶的資料切時間窗口
                segmenter = MultiWindowSegmenter()
                n_bands = len(train_bands)
                n_windows = segmenter.n_windows
                
                # 收集所有 (B, T) 組合的協方差
                train_covs_list = []
                test_covs_list = []
                
                # 使用 shrinkage 估計器 (oas 更穩定)
                cov_estimator = Covariances(estimator=DEFAULT_RIEMANNIAN.cov_estimator)
                
                for b_idx in range(n_bands):
                    # 對該頻帶切時間窗口
                    train_windows = segmenter.transform(train_bands[b_idx], sfreq)
                    test_windows = segmenter.transform(test_bands[b_idx], sfreq)
                    
                    for w_idx in range(n_windows):
                        # 計算協方差 (使用 pyriemann)
                        train_cov = cov_estimator.fit_transform(train_windows[w_idx])
                        test_cov = cov_estimator.transform(test_windows[w_idx])
                        train_covs_list.append(train_cov)
                        test_covs_list.append(test_cov)
                
                # 3. 粗篩 top-K (B, T) 組合
                n_combs = len(train_covs_list)
                comb_scores = []
                
                for comb_idx in range(n_combs):
                    covs = train_covs_list[comb_idx]
                    class0_mean = np.mean(covs[y_train == 0], axis=0)
                    class1_mean = np.mean(covs[y_train == 1], axis=0)
                    score = np.linalg.norm(class0_mean - class1_mean)
                    comb_scores.append(score)
                
                top_k = min(top_k_combinations, n_combs)
                top_indices = np.argsort(comb_scores)[-top_k:]
                
                # 4. 對選中的組合做切空間投影 (使用 pyriemann TangentSpace)
                train_features = []
                test_features = []
                
                for comb_idx in top_indices:
                    train_cov = train_covs_list[comb_idx]
                    test_cov = test_covs_list[comb_idx]
                    
                    # 使用 pyriemann TangentSpace (正確的黎曼均值 + 切空間投影)
                    ts = TangentSpace(metric='riemann')
                    train_tangent = ts.fit_transform(train_cov)  # 只用 train 計算 reference
                    test_tangent = ts.transform(test_cov)
                    
                    train_features.append(train_tangent)
                    test_features.append(test_tangent)
                
                # 串接特徵
                X_train_feat = np.hstack(train_features)
                X_test_feat = np.hstack(test_features)
                
                # 5. NCA + SVM
                clf = RiemannianSVMClassifier(
                    nca_components=nca_components,
                    nca_max_iter=DEFAULT_RIEMANNIAN.nca_max_iter,
                    svm_C=1.0,
                    svm_kernel='linear'
                )
                clf.fit(X_train_feat, y_train)
                y_pred = clf.predict(X_test_feat)
                
                fold_accuracies.append(accuracy_score(y_test, y_pred))
            
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            results[subject] = {
                'accuracy': mean_acc,
                'std': std_acc,
                'n_epochs': len(y)
            }
            
            # 更新進度條顯示準確率
            subject_pbar.set_postfix({
                "當前": subject,
                "準確率": f"{mean_acc*100:.1f}%"
            })
                
        except Exception as e:
            print(f"\n警告: {subject} 處理失敗 - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 計算平均
    if results:
        accuracies = [r['accuracy'] for r in results.values()]
        mean_overall = np.mean(accuracies)
        if verbose:
            print(f"\n{'='*40}")
            print(f"平均準確率: {mean_overall*100:.1f}%")
            print(f"{'='*40}")
        results['_summary'] = {
            'mean_accuracy': mean_overall,
            'std_accuracy': np.std(accuracies),
            'n_subjects': len(results)
        }
    
    return results


def _project_to_tangent(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    切空間投影
    
    Parameters
    ----------
    covs : np.ndarray
        協方差矩陣 (n_trials, Nc, Nc)
    reference : np.ndarray
        參考點 (Nc, Nc)
        
    Returns
    -------
    np.ndarray
        切空間特徵 (n_trials, d)
    """
    n_trials = covs.shape[0]
    n_channels = covs.shape[1]
    
    # 計算參考點的逆平方根
    eigvals, eigvecs = np.linalg.eigh(reference)
    eigvals = np.maximum(eigvals, 1e-10)
    ref_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    features = []
    for i in range(n_trials):
        # 白化
        whitened = ref_invsqrt @ covs[i] @ ref_invsqrt
        # 對數映射
        eigvals_w, eigvecs_w = np.linalg.eigh(whitened)
        eigvals_w = np.maximum(eigvals_w, 1e-10)
        log_whitened = eigvecs_w @ np.diag(np.log(eigvals_w)) @ eigvecs_w.T
        # 向量化 (上三角)
        idx = np.triu_indices(n_channels)
        vec = log_whitened[idx]
        features.append(vec)
    
    return np.array(features)


def main():
    """主程式"""
    print("\n" + "="*70)
    print("   黎曼空間 MI-BCI 分類方法")
    print("   論文: Study of MI-BCI classification method based on the")
    print("         Riemannian transform of personalized EEG spatiotemporal features")
    print("="*70)
    
    # 先只測試 BCI IV 2a
    results = run_riemannian_classification(
        dataset_name='BCICIV_2a',
        n_folds=5,
        top_k_combinations=50,
        nca_components=30,
        verbose=True
    )
    
    # 儲存結果
    from datetime import datetime
    results_dir = BASE_DIR / 'results_riemannian'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"BCICIV_2a_{timestamp}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("黎曼空間 MI-BCI 分類結果\n")
        f.write("="*50 + "\n\n")
        for subject, res in results.items():
            if not subject.startswith('_'):
                f.write(f"{subject}: {res['accuracy']*100:.1f}% (±{res['std']*100:.1f}%)\n")
        if '_summary' in results:
            f.write(f"\n平均: {results['_summary']['mean_accuracy']*100:.1f}%\n")
    
    print(f"\n結果已儲存至: {result_file}")
    
    return results


if __name__ == "__main__":
    main()
```

---

## preprocessing\__init__.py {#preprocessing-__init__-py}
**路徑**: `preprocessing\__init__.py`

```python
# -*- coding: utf-8 -*-
"""
Preprocessing 模組
"""
```

---

## preprocessing\filterbank.py {#preprocessing-filterbank-py}
**路徑**: `preprocessing\filterbank.py`

```python
# -*- coding: utf-8 -*-
"""
濾波器組 (Filter Bank) - 64 頻帶

頻率範圍: 4-40 Hz
頻帶寬度: 2, 4, 8, 16, 32 Hz
滑動步長: 2 Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple
import warnings


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple:
    """設計 Butterworth 帶通濾波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 確保頻率在有效範圍內
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                    fs: float, order: int = 5) -> np.ndarray:
    """
    帶通濾波
    
    Parameters
    ----------
    data : np.ndarray
        輸入資料 (n_channels, n_samples) 或 (n_trials, n_channels, n_samples)
    lowcut, highcut : float
        濾波頻率範圍
    fs : float
        取樣率
    order : int
        濾波器階數
        
    Returns
    -------
    np.ndarray
        濾波後資料
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if data.ndim == 2:
            return filtfilt(b, a, data, axis=1)
        elif data.ndim == 3:
            return filtfilt(b, a, data, axis=2)
        else:
            raise ValueError(f"資料維度必須為 2 或 3，但得到 {data.ndim}")


def generate_filter_bands() -> List[Tuple[float, float]]:
    """
    生成 64 個頻帶 (B1-B64)
    
    頻率範圍: 4-40 Hz
    頻帶寬度: 2, 4, 8, 16, 32 Hz
    滑動步長: 2 Hz
    
    Returns
    -------
    List[Tuple[float, float]]
        64 個頻帶列表
    """
    bands = []
    for bandwidth in [2, 4, 8, 16, 32]:
        for start in range(4, 40, 2):
            if start + bandwidth <= 40:
                bands.append((float(start), float(start + bandwidth)))
    return bands


class FilterBank:
    """
    濾波器組
    
    對 EEG 資料應用 64 個頻帶濾波
    """
    
    def __init__(self, bands: List[Tuple[float, float]] = None, order: int = 5):
        """
        初始化
        
        Parameters
        ----------
        bands : List[Tuple[float, float]], optional
            頻帶列表，預設使用 B1-B64
        order : int
            濾波器階數
        """
        self.bands = bands if bands is not None else generate_filter_bands()
        self.order = order
        self.n_bands = len(self.bands)
    
    def transform(self, X: np.ndarray, sfreq: float) -> List[np.ndarray]:
        """
        應用濾波器組
        
        Parameters
        ----------
        X : np.ndarray
            EEG 資料 (n_trials, n_channels, n_samples)
        sfreq : float
            取樣率
            
        Returns
        -------
        List[np.ndarray]
            濾波後資料列表，每個元素形狀與輸入相同
        """
        filtered = []
        for low, high in self.bands:
            try:
                filtered_data = bandpass_filter(X, low, high, sfreq, self.order)
                filtered.append(filtered_data)
            except Exception as e:
                # 如果濾波失敗，使用原始資料
                warnings.warn(f"濾波失敗 ({low}-{high} Hz): {e}")
                filtered.append(X.copy())
        return filtered
    
    def get_band_info(self) -> List[str]:
        """取得頻帶資訊字串"""
        info = []
        for i, (low, high) in enumerate(self.bands):
            info.append(f"B{i+1}: {low:.0f}-{high:.0f} Hz")
        return info


def apply_filterbank_to_windows(
    window_data: List[np.ndarray],
    sfreq: float,
    bands: List[Tuple[float, float]] = None,
    order: int = 5
) -> np.ndarray:
    """
    對所有時間窗口應用濾波器組
    
    Parameters
    ----------
    window_data : List[np.ndarray]
        時間窗口資料列表，每個元素形狀 (n_trials, n_channels, n_samples_w)
    sfreq : float
        取樣率
    bands : List[Tuple[float, float]], optional
        頻帶列表
    order : int
        濾波器階數
        
    Returns
    -------
    np.ndarray
        形狀 (n_trials, n_windows, n_bands, n_channels, n_samples_min)
        或以 list of list 形式返回
    """
    if bands is None:
        bands = generate_filter_bands()
    
    fb = FilterBank(bands=bands, order=order)
    n_windows = len(window_data)
    n_bands = len(bands)
    
    # 對每個時間窗口應用濾波器組
    # 結果: window_band_data[w][b] = (n_trials, n_channels, n_samples)
    all_filtered = []
    
    for w_idx, w_data in enumerate(window_data):
        band_filtered = fb.transform(w_data, sfreq)
        all_filtered.append(band_filtered)
    
    return all_filtered  # List[List[np.ndarray]]
```

---

## preprocessing\multiwindow.py {#preprocessing-multiwindow-py}
**路徑**: `preprocessing\multiwindow.py`

```python
# -*- coding: utf-8 -*-
"""
多時間窗口切分

將 EEG 資料按 T1-T8 時間窗口切分
"""

import numpy as np
from typing import List, Tuple


def segment_time_windows(
    data: np.ndarray,
    sfreq: float,
    windows: List[Tuple[float, float]],
    epoch_tmin: float = 0.0
) -> List[np.ndarray]:
    """
    將 EEG 資料按多個時間窗口切分
    
    Parameters
    ----------
    data : np.ndarray
        EEG 資料 (n_trials, n_channels, n_samples)
    sfreq : float
        取樣率 (Hz)
    windows : List[Tuple[float, float]]
        時間窗口列表 [(tmin1, tmax1), (tmin2, tmax2), ...]
        時間相對於 epoch_tmin
    epoch_tmin : float
        Epoch 的起始時間 (秒)，預設 0.0
        
    Returns
    -------
    List[np.ndarray]
        切分後的資料列表，每個元素形狀為 (n_trials, n_channels, n_window_samples)
    """
    segmented = []
    
    for tmin, tmax in windows:
        # 計算 sample 索引
        start_sample = int((tmin - epoch_tmin) * sfreq)
        end_sample = int((tmax - epoch_tmin) * sfreq)
        
        # 確保索引在有效範圍內
        start_sample = max(0, start_sample)
        end_sample = min(data.shape[2], end_sample)
        
        # 切分
        segment = data[:, :, start_sample:end_sample]
        segmented.append(segment)
    
    return segmented


def get_default_windows() -> List[Tuple[float, float]]:
    """
    取得預設的 T1-T8 時間窗口
    
    Returns
    -------
    List[Tuple[float, float]]
        T1-T8 時間窗口列表
    """
    return [
        (0.0, 1.0),   # T1
        (0.5, 1.5),   # T2
        (1.0, 2.0),   # T3
        (1.5, 2.5),   # T4
        (2.0, 3.0),   # T5
        (2.5, 3.5),   # T6
        (0.5, 2.5),   # T7 - MI 任務期間
        (0.0, 4.0),   # T8 - 完整期間
    ]


class MultiWindowSegmenter:
    """
    多時間窗口切分器
    
    將 EEG 資料按 T1-T8 時間窗口切分
    """
    
    def __init__(self, windows: List[Tuple[float, float]] = None, epoch_tmin: float = 0.0):
        """
        初始化
        
        Parameters
        ----------
        windows : List[Tuple[float, float]], optional
            時間窗口列表，預設使用 T1-T8
        epoch_tmin : float
            Epoch 起始時間
        """
        self.windows = windows if windows is not None else get_default_windows()
        self.epoch_tmin = epoch_tmin
        self.n_windows = len(self.windows)
    
    def transform(self, X: np.ndarray, sfreq: float) -> List[np.ndarray]:
        """
        切分資料
        
        Parameters
        ----------
        X : np.ndarray
            EEG 資料 (n_trials, n_channels, n_samples)
        sfreq : float
            取樣率
            
        Returns
        -------
        List[np.ndarray]
            切分後的資料列表
        """
        return segment_time_windows(X, sfreq, self.windows, self.epoch_tmin)
    
    def get_window_info(self) -> List[str]:
        """取得時間窗口資訊字串"""
        info = []
        for i, (tmin, tmax) in enumerate(self.windows):
            info.append(f"T{i+1}: {tmin:.1f}s - {tmax:.1f}s")
        return info
```

---

## utils\__init__.py {#utils-__init__-py}
**路徑**: `utils\__init__.py`

```python
# -*- coding: utf-8 -*-
"""
Utils 模組
"""
```

---

## utils\cache.py {#utils-cache-py}
**路徑**: `utils\cache.py`

```python
# -*- coding: utf-8 -*-
"""
快取工具

使用 joblib 快取濾波結果以加速運算
"""

import os
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any
import warnings

try:
    from joblib import Memory
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib 未安裝，快取功能將停用")


def get_cache_dir(base_dir: Path = None) -> Path:
    """取得快取目錄"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / ".cache_riemannian"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_memory(cache_dir: Path = None) -> Optional['Memory']:
    """取得 joblib Memory 物件"""
    if not JOBLIB_AVAILABLE:
        return None
    
    cache_dir = get_cache_dir(cache_dir)
    return Memory(location=str(cache_dir), verbose=0)


def cached_function(func: Callable, cache_dir: Path = None) -> Callable:
    """
    裝飾器：快取函數結果
    
    Parameters
    ----------
    func : Callable
        要快取的函數
    cache_dir : Path, optional
        快取目錄
        
    Returns
    -------
    Callable
        快取後的函數
    """
    memory = get_memory(cache_dir)
    if memory is not None:
        return memory.cache(func)
    else:
        return func


def compute_data_hash(data: np.ndarray) -> str:
    """計算資料的 hash 值"""
    return hashlib.md5(data.tobytes()).hexdigest()[:16]


class FilterCache:
    """
    濾波結果快取
    
    用於快取耗時的濾波運算結果
    """
    
    def __init__(self, cache_dir: Path = None, enabled: bool = True):
        """
        初始化
        
        Parameters
        ----------
        cache_dir : Path, optional
            快取目錄
        enabled : bool
            是否啟用快取
        """
        self.enabled = enabled and JOBLIB_AVAILABLE
        self.cache_dir = get_cache_dir(cache_dir)
        self._cache = {}
    
    def get_or_compute(
        self, 
        key: str, 
        compute_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """
        從快取取得結果，若無則計算
        
        Parameters
        ----------
        key : str
            快取鍵值
        compute_func : Callable
            計算函數
        *args, **kwargs
            傳遞給計算函數的參數
            
        Returns
        -------
        Any
            計算結果
        """
        if not self.enabled:
            return compute_func(*args, **kwargs)
        
        # 檢查記憶體快取
        if key in self._cache:
            return self._cache[key]
        
        # 檢查磁碟快取
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                result = np.load(cache_file, allow_pickle=True)
                self._cache[key] = result
                return result
            except Exception:
                pass
        
        # 計算並快取
        result = compute_func(*args, **kwargs)
        
        # 儲存到記憶體
        self._cache[key] = result
        
        # 儲存到磁碟 (僅限 numpy array)
        if isinstance(result, np.ndarray):
            try:
                np.save(cache_file, result)
            except Exception:
                pass
        
        return result
    
    def clear(self):
        """清除快取"""
        self._cache.clear()
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npy"):
                try:
                    f.unlink()
                except Exception:
                    pass
```

---
