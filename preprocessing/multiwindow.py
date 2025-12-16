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
