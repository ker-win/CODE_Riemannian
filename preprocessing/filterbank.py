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
