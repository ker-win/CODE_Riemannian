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
                
                # 1. 多時間窗口切分
                segmenter = MultiWindowSegmenter()
                train_windows = segmenter.transform(X_train, sfreq)
                test_windows = segmenter.transform(X_test, sfreq)
                
                # 2. 濾波器組 (對每個時間窗口)
                train_filtered = apply_filterbank_to_windows(train_windows, sfreq, FILTER_BANDS)
                test_filtered = apply_filterbank_to_windows(test_windows, sfreq, FILTER_BANDS)
                
                # 3. 計算協方差矩陣 (對每個 T,B 組合)
                n_windows = len(train_filtered)
                n_bands = len(train_filtered[0]) if n_windows > 0 else 0
                
                train_covs_list = []
                test_covs_list = []
                
                for w_idx in range(n_windows):
                    for b_idx in range(n_bands):
                        train_cov = compute_covariance(train_filtered[w_idx][b_idx])
                        test_cov = compute_covariance(test_filtered[w_idx][b_idx])
                        train_covs_list.append(train_cov)
                        test_covs_list.append(test_cov)
                
                # 4. 粗篩 top-K (T,B) 組合
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
                
                # 5. 對選中的組合做切空間投影
                train_features = []
                test_features = []
                
                for comb_idx in top_indices:
                    train_cov = train_covs_list[comb_idx]
                    test_cov = test_covs_list[comb_idx]
                    
                    ref = np.mean(train_cov, axis=0)
                    ref = ref + 1e-6 * np.eye(ref.shape[0])
                    
                    train_tangent = _project_to_tangent(train_cov, ref)
                    test_tangent = _project_to_tangent(test_cov, ref)
                    
                    train_features.append(train_tangent)
                    test_features.append(test_tangent)
                
                # 串接特徵
                X_train_feat = np.hstack(train_features)
                X_test_feat = np.hstack(test_features)
                
                # 6. NCA + SVM
                clf = RiemannianSVMClassifier(
                    nca_components=nca_components,
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
