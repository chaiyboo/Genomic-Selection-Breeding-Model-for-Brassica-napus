#!/usr/bin/env python3
"""
Random Forest model with 10-fold cross validation for genomic prediction
Fixed version - with correlation analysis
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import joblib
import warnings
from scipy import stats  # 添加scipy用于相关性分析
warnings.filterwarnings('ignore')

# 设置路径
BASE_DIR = "/home_song/ybchai/nam"
GENO_FILE = f"{BASE_DIR}/geno/NAM_glnexus_reheader_plink.vcf.gz"
PHENO_DIR = f"{BASE_DIR}/pheno_BLUP"
MODEL_DIR = f"{BASE_DIR}/model"
OUTPUT_DIR = f"{BASE_DIR}/model/rf-sklearn"
FOLD_LIST_DIR = f"{BASE_DIR}/10fold-list"
TRAITS = ["flower_win", "flower_spr", "weight", "oil"]

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)  # 新增plots目录

def calculate_correlation(y_true, y_pred):
    """
    计算Pearson和Spearman相关系数
    """
    # Pearson相关系数
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Spearman秩相关系数
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }

def read_plink_raw(raw_file):
    """
    读取PLINK的raw格式文件
    """
    print(f"Reading genotype data from {raw_file}")
    
    # 读取raw文件
    df = pd.read_csv(raw_file, sep=' ', header=0, low_memory=False)
    
    # 提取基因型数据（从第7列开始是基因型）
    genotype_df = df.iloc[:, 6:]
    
    # 提取个体ID - 只使用IID
    sample_ids = df['IID'].astype(str)
    
    print(f"Loaded {genotype_df.shape[1]} markers for {genotype_df.shape[0]} samples")
    print(f"First few sample IDs: {sample_ids[:5].tolist()}")
    
    return genotype_df, sample_ids

def read_phenotype(pheno_file, trait):
    """
    读取表型数据
    """
    print(f"Reading phenotype data for {trait} from {pheno_file}")
    
    if not os.path.exists(pheno_file):
        print(f"Warning: Phenotype file {pheno_file} not found")
        return None
    
    # 读取表型文件
    df = pd.read_csv(pheno_file, sep='\s+', header=0)
    
    # 提取ID和表型值
    sample_ids = df.iloc[:, 0].astype(str)
    phenotype = df.iloc[:, 1].values
    
    print(f"  Found {len(sample_ids)} samples with phenotype")
    print(f"  First few IDs: {sample_ids[:5].tolist()}")
    print(f"  First few values: {phenotype[:5]}")
    
    return pd.Series(phenotype, index=sample_ids)

def read_fold_ids(fold_file):
    """
    读取折的个体ID
    """
    ids = []
    with open(fold_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                ids.append(parts[0])
    
    return ids

def prepare_data_for_fold(genotype_df, phenotype_series, train_ids, test_ids, trait, fold):
    """
    准备特定折的训练和测试数据
    """
    # 找到共同的个体
    common_samples = genotype_df.index.intersection(phenotype_series.index)
    print(f"  Common samples between genotype and phenotype: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print(f"  ERROR: No common samples found!")
        return None, None, None, None, None, None
    
    # 筛选训练集和测试集的共同个体
    train_samples = [s for s in common_samples if s in train_ids]
    test_samples = [s for s in common_samples if s in test_ids]
    
    print(f"  Training samples in common: {len(train_samples)}")
    print(f"  Testing samples in common: {len(test_samples)}")
    
    if len(train_samples) == 0 or len(test_samples) == 0:
        return None, None, None, None, None, None
    
    # 准备训练数据
    X_train = genotype_df.loc[train_samples].values
    y_train = phenotype_series.loc[train_samples].values
    
    # 准备测试数据
    X_test = genotype_df.loc[test_samples].values
    y_test = phenotype_series.loc[test_samples].values
    
    return X_train, y_train, X_test, y_test, train_samples, test_samples

def train_random_forest(X_train, y_train, X_test, y_test, trait, fold):
    """
    训练随机森林模型并进行预测
    """
    print(f"  Training Random Forest model...")
    print(f"    Training set size: {X_train.shape}")
    print(f"    Test set size: {X_test.shape}")
    
    # 参数分布
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    # 创建基础模型
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0,
        bootstrap=True,
        oob_score=True
    )
    
    # 使用随机搜索
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    # 训练
    random_search.fit(X_train, y_train)
    
    # 最佳模型
    best_model = random_search.best_estimator_
    
    print(f"    Best parameters: {random_search.best_params_}")
    if hasattr(best_model, 'oob_score_'):
        print(f"    OOB Score: {best_model.oob_score_:.4f}")
    
    # 预测
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # 计算基本指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 计算相关性
    train_corr = calculate_correlation(y_train, y_train_pred)
    test_corr = calculate_correlation(y_test, y_test_pred)
    
    print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"    Test Pearson r: {test_corr['pearson_r']:.4f} (p={test_corr['pearson_p']:.4f})")
    print(f"    Test Spearman r: {test_corr['spearman_r']:.4f} (p={test_corr['spearman_p']:.4f})")
    
    # 整合所有指标
    metrics = {
        'trait': trait,
        'fold': fold,
        # 基本指标
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        # 训练集相关性
        'train_pearson_r': train_corr['pearson_r'],
        'train_pearson_p': train_corr['pearson_p'],
        'train_spearman_r': train_corr['spearman_r'],
        'train_spearman_p': train_corr['spearman_p'],
        # 测试集相关性
        'test_pearson_r': test_corr['pearson_r'],
        'test_pearson_p': test_corr['pearson_p'],
        'test_spearman_r': test_corr['spearman_r'],
        'test_spearman_p': test_corr['spearman_p'],
        # 其他信息
        'oob_score': best_model.oob_score_ if hasattr(best_model, 'oob_score_') else np.nan,
        'best_params': str(random_search.best_params_),
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test)
    }
    
    return best_model, y_train_pred, y_test_pred, metrics

def plot_fold_results(y_true, y_pred, trait, fold, output_dir):
    """
    为每个fold创建预测vs实际值的散点图
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        
        # 散点图
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # 添加对角线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
        
        # 计算相关系数
        corr = calculate_correlation(y_true, y_pred)
        
        # 添加回归线
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot([min_val, max_val], p([min_val, max_val]), 'b-', alpha=0.5, label='Regression line')
        
        # 添加标题和标签
        plt.xlabel('Observed Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{trait} - Fold {fold}\nPearson r = {corr["pearson_r"]:.3f} (p={corr["pearson_p"]:.3f})\nSpearman r = {corr["spearman_r"]:.3f} (p={corr["spearman_p"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        plot_file = f"{output_dir}/plots/{trait}_fold{fold}_scatter.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Plot saved to {plot_file}")
        
    except Exception as e:
        print(f"    Warning: Could not create plot: {e}")

def main():
    """
    主函数
    """
    print("="*60)
    print("Random Forest with 10-fold Cross Validation")
    print("="*60)
    
    # 1. 读取基因型数据
    print("\nStep 1: Loading genotype data...")
    raw_file = f"{MODEL_DIR}/NAM_gblup.pruned.filled.raw"
    
    if not os.path.exists(raw_file):
        print(f"ERROR: Genotype file not found: {raw_file}")
        return
    
    genotype_df, sample_ids = read_plink_raw(raw_file)
    genotype_df.index = sample_ids
    
    print(f"Genotype data shape: {genotype_df.shape}")
    
    # 2. 读取所有折的ID
    print("\nStep 2: Loading fold IDs...")
    fold_ids = {}
    all_fold_ids = set()
    
    for fold in range(1, 11):
        fold_file = f"{FOLD_LIST_DIR}/group{fold}_ids-FIID.txt"
        ids = read_fold_ids(fold_file)
        fold_ids[fold] = set(ids)
        all_fold_ids.update(ids)
        print(f"  Fold {fold}: {len(ids)} IDs")
    
    print(f"Total unique IDs across all folds: {len(all_fold_ids)}")
    
    # 3. 对每个性状进行分析
    all_summaries = []
    
    for trait in TRAITS:
        print(f"\n{'='*50}")
        print(f"Processing trait: {trait}")
        print('='*50)
        
        # 读取表型数据
        pheno_file = f"{PHENO_DIR}/{trait}_BLUP.txt"
        phenotype_series = read_phenotype(pheno_file, trait)
        
        if phenotype_series is None:
            print(f"Skipping {trait} due to missing phenotype file")
            continue
        
        # 检查ID匹配情况
        common_with_geno = set(genotype_df.index).intersection(set(phenotype_series.index))
        print(f"Samples with both genotype and phenotype: {len(common_with_geno)}")
        
        common_with_folds = common_with_geno.intersection(all_fold_ids)
        print(f"Samples with genotype, phenotype, and fold assignment: {len(common_with_folds)}")
        
        if len(common_with_folds) == 0:
            print(f"ERROR: No samples with complete data for {trait}!")
            continue
        
        # 4. 10折交叉验证
        all_metrics = []
        
        for fold in range(1, 11):
            print(f"\n  Fold {fold}/10")
            
            # 训练集是其他9折
            train_ids = set()
            for i in range(1, 11):
                if i != fold:
                    train_ids.update(fold_ids[i])
            
            # 测试集是当前折
            test_ids = fold_ids[fold]
            
            print(f"    Train IDs: {len(train_ids)}")
            print(f"    Test IDs: {len(test_ids)}")
            
            # 准备数据
            result = prepare_data_for_fold(
                genotype_df, phenotype_series, train_ids, test_ids, trait, fold
            )
            
            if result[0] is None:
                print(f"    Skipping fold {fold} due to no samples")
                continue
            
            X_train, y_train, X_test, y_test, train_samples, test_samples = result
            
            # 训练模型
            model, y_train_pred, y_test_pred, metrics = train_random_forest(
                X_train, y_train, X_test, y_test, trait, fold
            )
            
            # 保存模型
            model_file = f"{OUTPUT_DIR}/models/{trait}_fold{fold}_rf.pkl"
            joblib.dump(model, model_file)
            
            # 保存预测结果（包含所有信息）
            pred_df = pd.DataFrame({
                'sample_id': list(train_samples) + list(test_samples),
                'actual': list(y_train) + list(y_test),
                'predicted': list(y_train_pred) + list(y_test_pred),
                'fold': ['train']*len(train_samples) + ['test']*len(test_samples),
                'residual': (list(y_train - y_train_pred) + list(y_test - y_test_pred))
            })
            pred_file = f"{OUTPUT_DIR}/predictions/{trait}_fold{fold}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            
            # 创建散点图
            plot_fold_results(y_test, y_test_pred, trait, fold, OUTPUT_DIR)
            
            # 保存指标
            metrics_df = pd.DataFrame([metrics])
            metrics_file = f"{OUTPUT_DIR}/metrics/{trait}_fold{fold}_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            
            all_metrics.append(metrics_df)
        
        # 5. 汇总所有折的结果
        if all_metrics:
            all_metrics_df = pd.concat(all_metrics, ignore_index=True)
            
            summary_metrics = {
                'trait': trait,
                'n_folds_completed': len(all_metrics),
                # R²
                'avg_test_r2': all_metrics_df['test_r2'].mean(),
                'std_test_r2': all_metrics_df['test_r2'].std(),
                # RMSE
                'avg_test_rmse': all_metrics_df['test_rmse'].mean(),
                'std_test_rmse': all_metrics_df['test_rmse'].std(),
                # MAE
                'avg_test_mae': all_metrics_df['test_mae'].mean(),
                'std_test_mae': all_metrics_df['test_mae'].std(),
                # Pearson correlation
                'avg_test_pearson_r': all_metrics_df['test_pearson_r'].mean(),
                'std_test_pearson_r': all_metrics_df['test_pearson_r'].std(),
                'avg_test_pearson_p': all_metrics_df['test_pearson_p'].mean(),
                # Spearman correlation
                'avg_test_spearman_r': all_metrics_df['test_spearman_r'].mean(),
                'std_test_spearman_r': all_metrics_df['test_spearman_r'].std(),
                'avg_test_spearman_p': all_metrics_df['test_spearman_p'].mean(),
                # OOB score
                'avg_oob_score': all_metrics_df['oob_score'].mean(),
                'std_oob_score': all_metrics_df['oob_score'].std()
            }
            
            summary_df = pd.DataFrame([summary_metrics])
            summary_file = f"{OUTPUT_DIR}/metrics/{trait}_summary_metrics.csv"
            summary_df.to_csv(summary_file, index=False)
            all_summaries.append(summary_df)
            
            print(f"\n  Results for {trait}:")
            print(f"    Completed folds: {summary_metrics['n_folds_completed']}/10")
            print(f"    Test R²: {summary_metrics['avg_test_r2']:.4f} ± {summary_metrics['std_test_r2']:.4f}")
            print(f"    Test Pearson r: {summary_metrics['avg_test_pearson_r']:.4f} ± {summary_metrics['std_test_pearson_r']:.4f}")
            print(f"    Test Spearman r: {summary_metrics['avg_test_spearman_r']:.4f} ± {summary_metrics['std_test_spearman_r']:.4f}")
            print(f"    Test RMSE: {summary_metrics['avg_test_rmse']:.4f} ± {summary_metrics['std_test_rmse']:.4f}")
    
    # 6. 创建所有性状的综合比较图
    if all_summaries:
        try:
            import matplotlib.pyplot as plt
            
            summary_df = pd.concat(all_summaries, ignore_index=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # R²比较
            ax = axes[0, 0]
            x = np.arange(len(summary_df))
            ax.bar(x, summary_df['avg_test_r2'], yerr=summary_df['std_test_r2'], 
                   capsize=5, color='steelblue', alpha=0.7)
            ax.set_xlabel('Trait')
            ax.set_ylabel('R²')
            ax.set_title('Test R² by Trait')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['trait'])
            ax.grid(True, alpha=0.3)
            
            # Pearson相关性
            ax = axes[0, 1]
            ax.bar(x, summary_df['avg_test_pearson_r'], yerr=summary_df['std_test_pearson_r'],
                   capsize=5, color='coral', alpha=0.7)
            ax.set_xlabel('Trait')
            ax.set_ylabel('Pearson r')
            ax.set_title('Test Pearson Correlation by Trait')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['trait'])
            ax.grid(True, alpha=0.3)
            
            # Spearman相关性
            ax = axes[1, 0]
            ax.bar(x, summary_df['avg_test_spearman_r'], yerr=summary_df['std_test_spearman_r'],
                   capsize=5, color='seagreen', alpha=0.7)
            ax.set_xlabel('Trait')
            ax.set_ylabel('Spearman r')
            ax.set_title('Test Spearman Correlation by Trait')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['trait'])
            ax.grid(True, alpha=0.3)
            
            # RMSE
            ax = axes[1, 1]
            ax.bar(x, summary_df['avg_test_rmse'], yerr=summary_df['std_test_rmse'],
                   capsize=5, color='purple', alpha=0.7)
            ax.set_xlabel('Trait')
            ax.set_ylabel('RMSE')
            ax.set_title('Test RMSE by Trait')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['trait'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/plots/all_traits_comparison.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{OUTPUT_DIR}/plots/all_traits_comparison.pdf", bbox_inches='tight')
            print(f"\nComparison plot saved to {OUTPUT_DIR}/plots/all_traits_comparison.png")
            
        except Exception as e:
            print(f"Warning: Could not create comparison plot: {e}")
    
    print(f"\n{'='*50}")
    print("Analysis completed!")
    print(f"All results saved to {OUTPUT_DIR}")
    print('='*50)

if __name__ == "__main__":
    main()