#!/usr/bin/env python3
import os
import sys
import argparse
import random

# =============================
# 固定随机种子（可复现）
# =============================
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
np.random.seed(SEED)

# =============================
# 无 GUI 服务器画图安全设置
# =============================
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(SEED)

# =============================
# 参数
# =============================
parser = argparse.ArgumentParser(description="DeepGS 10-fold CV")
parser.add_argument("--trait", required=True, help="Trait name, e.g. flower_win")
parser.add_argument("--geno", required=True, help="Genotype .raw file (filled)")
args = parser.parse_args()

trait = args.trait
geno_file = args.geno

pheno_file = f"/home_song/ybchai/nam/pheno_BLUP/{trait}_BLUP.txt"
fold_dir = "/home_song/ybchai/nam/10fold-list"
out_dir = f"/home_song/ybchai/nam/module/deepgs/{trait}"

os.makedirs(out_dir, exist_ok=True)

# =============================
# 文件存在性检查
# =============================
assert os.path.exists(geno_file), f"Genotype file not found: {geno_file}"
assert os.path.exists(pheno_file), f"Phenotype file not found: {pheno_file}"
assert os.path.exists(fold_dir), f"Fold dir not found: {fold_dir}"

# =============================
# 1. 读取基因型
# =============================
geno = pd.read_csv(geno_file, sep=r"\s+")

required_cols = {"FID", "IID"}
assert required_cols.issubset(geno.columns), "Missing FID/IID columns in .raw file"

ids = geno["IID"].astype(str).values
all_ids = set(ids)

X_raw = geno.drop(
    columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"],
    errors="ignore"
).values.astype(float)

print("Genotype matrix shape:", X_raw.shape)
print("Initial NaN count:", np.isnan(X_raw).sum())

# =============================
# 2. 读取表型（严格按 IID 对齐）
# =============================
pheno = pd.read_csv(
    pheno_file,
    sep=r"\s+",
    header=None,
    names=["ID", "BLUP"]
)

pheno["ID"] = pheno["ID"].astype(str)
pheno = pheno.set_index("ID")

common_ids = np.intersect1d(ids, pheno.index.values)
assert len(common_ids) > 0, "No overlapping IDs between genotype and phenotype"

keep_mask = np.isin(ids, common_ids)

ids = ids[keep_mask]
X_raw = X_raw[keep_mask, :]
y = pheno.loc[ids, "BLUP"].values.astype(float)

all_ids = set(ids)

print("Samples after ID matching:", len(ids))

# =============================
# 3. 十折 CV
# =============================
all_pred = []
all_obs = []
fold_r = []

for fold in range(1, 11):

    print(f"\n========== Fold {fold} ==========")

    fold_file = f"{fold_dir}/group{fold}_ids-FIID.txt"
    assert os.path.exists(fold_file), f"Missing fold file: {fold_file}"

    # ---- 读取 fold ID，只取第一列 ----
    test_ids = set(
        pd.read_csv(
            fold_file,
            sep=r"\s+",
            header=None,
            usecols=[0]
        )[0].astype(str)
    )

    # ---- 关键修复：fold 只保留 raw 中存在的 ID ----
    test_ids = test_ids & all_ids

    test_mask = np.isin(ids, list(test_ids))
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        raise ValueError(f"Fold {fold}: no test samples after ID matching")

    if train_mask.sum() == 0:
        raise ValueError(f"Fold {fold}: no train samples after ID matching")

    print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")

    X_train_raw = X_raw[train_mask]
    X_test_raw  = X_raw[test_mask]

    y_train = y[train_mask]
    y_test  = y[test_mask]

    # =============================
    # 4. 训练集预处理（fit 只在 train）
    # =============================
    valid_snp_mask = ~np.all(np.isnan(X_train_raw), axis=0)

    X_train_raw2 = X_train_raw[:, valid_snp_mask]
    X_test_raw2  = X_test_raw[:, valid_snp_mask]

    imputer = SimpleImputer(strategy="mean")
    var_filter = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()
    pca = PCA(n_components=300, random_state=SEED)

    X_train = imputer.fit_transform(X_train_raw2)
    X_train = var_filter.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)

    print("Removed all-NaN SNPs:",
          X_train_raw.shape[1] - X_train_raw2.shape[1])
    print("Removed zero-variance SNPs:",
          X_train_raw2.shape[1] - var_filter.get_support().sum())
    print("PCA features:", X_train.shape[1])

    # =============================
    # 5. 测试集 transform
    # =============================
    X_test = imputer.transform(X_test_raw2)
    X_test = var_filter.transform(X_test)
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)

    # =============================
    # 6. DeepGS 模型
    # =============================
    model = Sequential([
        Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=20, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).ravel()

    r, _ = pearsonr(y_test, y_pred)
    fold_r.append(r)

    print(f"Fold {fold} Pearson r = {r:.4f}")

    fold_res = pd.DataFrame({
        "ID": ids[test_mask],
        "Observed": y_test,
        "Predicted": y_pred,
        "Fold": fold
    })

    fold_res.to_csv(
        f"{out_dir}/fold{fold}_prediction.txt",
        sep="\t",
        index=False
    )

    all_pred.extend(y_pred.tolist())
    all_obs.extend(y_test.tolist())

# =============================
# 7. 总体结果
# =============================
r_all, _ = pearsonr(all_obs, all_pred)

with open(f"{out_dir}/deepgs_cv_summary.txt", "w") as f:
    f.write(f"Trait: {trait}\n")
    f.write(f"Total samples: {len(all_obs)}\n")
    f.write(f"10-fold DeepGS correlation: {r_all:.4f}\n")
    f.write("Fold-wise r:\n")
    for i, r in enumerate(fold_r, 1):
        f.write(f"  Fold {i}: {r:.4f}\n")

print("\nFinished DeepGS:", trait)
print("Overall Pearson r =", round(r_all, 4))
