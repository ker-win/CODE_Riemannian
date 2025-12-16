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
