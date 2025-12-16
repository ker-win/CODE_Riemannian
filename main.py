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


def _stageA_score_riemann(covs: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """
    Stage A: 使用黎曼距離計算 (T,B) 組合的可分性分數
    
    score = between_class_distance / (within_class_distance + eps)
    
    Parameters
    ----------
    covs : np.ndarray
        協方差矩陣 (n_trials, Nc, Nc)
    y : np.ndarray
        標籤
    eps : float
        避免除以零
        
    Returns
    -------
    float
        可分性分數 (越高越好)
    """
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.distance import distance_riemann
    
    classes = np.unique(y)
    
    # 計算每個類別的黎曼均值和類內距離
    means = []
    within = 0.0
    
    for c in classes:
        cov_c = covs[y == c]
        if len(cov_c) < 2:
            # 樣本太少，跳過
            continue
        try:
            m_c = mean_riemann(cov_c)
            means.append(m_c)
            # 類內距離：樣本到類均值的平均黎曼距離
            dists = [distance_riemann(cov_c[i], m_c) for i in range(len(cov_c))]
            within += np.mean(dists)
        except Exception:
            # 計算失敗時使用歐式距離
            m_c = np.mean(cov_c, axis=0)
            means.append(m_c)
            within += np.mean([np.linalg.norm(cov_c[i] - m_c) for i in range(len(cov_c))])
    
    if len(means) < 2:
        return 0.0
    
    within /= len(classes)
    
    # 類間距離：類均值之間的平均黎曼距離
    between_vals = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            try:
                between_vals.append(distance_riemann(means[i], means[j]))
            except Exception:
                between_vals.append(np.linalg.norm(means[i] - means[j]))
    
    between = float(np.mean(between_vals)) if between_vals else 0.0
    
    return between / (within + eps)

def run_riemannian_classification(
    dataset_name: str = 'BCICIV_2a',
    n_folds: int = 5,
    top_k_combinations: int = None,
    nca_components: int = None,
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
    top_k_combinations : int, optional
        選取的 (T,B) 組合數量，預設從 config 讀取
    nca_components : int, optional
        NCA 降維維度，預設從 config 讀取
    verbose : bool
        是否輸出詳細資訊
        
    Returns
    -------
    dict
        包含每位受試者結果
    """
    # 從 config 讀取預設值
    if top_k_combinations is None:
        top_k_combinations = DEFAULT_RIEMANNIAN.top_k_combinations
    if nca_components is None:
        nca_components = DEFAULT_RIEMANNIAN.nca_n_components
    
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
                
                # 3. 粗篩 top-K (B, T) 組合 (使用黎曼距離)
                from pyriemann.utils.mean import mean_riemann
                from pyriemann.utils.distance import distance_riemann
                
                n_combs = len(train_covs_list)
                comb_scores = []
                
                for comb_idx in range(n_combs):
                    covs = train_covs_list[comb_idx]
                    # 使用黎曼距離計算可分性分數
                    score = _stageA_score_riemann(covs, y_train)
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
                
                # 5. NCA + SVM (with grid search for C)
                from sklearn.model_selection import GridSearchCV
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                from sklearn.neighbors import NeighborhoodComponentsAnalysis
                
                # 建立 Pipeline
                n_comp = min(nca_components, X_train_feat.shape[1] - 1, X_train_feat.shape[0] - 1)
                n_comp = max(1, n_comp)
                
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('nca', NeighborhoodComponentsAnalysis(
                        n_components=n_comp,
                        max_iter=DEFAULT_RIEMANNIAN.nca_max_iter,
                        random_state=42
                    )),
                    ('svm', SVC(kernel='linear', random_state=42))
                ])
                
                # Grid search for SVM C
                param_grid = {'svm__C': [0.01, 0.1, 1, 10, 100]}
                grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
                grid.fit(X_train_feat, y_train)
                
                y_pred = grid.predict(X_test_feat)
                
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
