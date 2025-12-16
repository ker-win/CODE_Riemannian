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
    
    def __init__(self, n_components: int = 30, random_state: int = 42):
        """
        初始化
        
        Parameters
        ----------
        n_components : int
            NCA 降維後的維度
        random_state : int
            隨機種子
        """
        self.n_components = n_components
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
            max_iter=100
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
        nca_components: int = 30,
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
        svm_C : float
            SVM 正則化參數
        svm_kernel : str
            SVM kernel 類型
        random_state : int
            隨機種子
        """
        self.nca_components = nca_components
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
