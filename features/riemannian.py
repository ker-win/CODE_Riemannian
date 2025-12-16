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
