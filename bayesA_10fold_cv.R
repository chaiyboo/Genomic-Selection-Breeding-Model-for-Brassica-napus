#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(BGLR)
})

args <- commandArgs(trailingOnly = TRUE)
trait <- args[1]

cat("Running BayesA for trait:", trait, "\n")

# =============================
# 路径
# =============================
geno_file <- "/home_song/ybchai/nam/module/NAM_gblup.pruned.raw"
pheno_file <- paste0("/home_song/ybchai/nam/pheno_BLUP/", trait, "_BLUP.txt")
fold_dir <- "/home_song/ybchai/nam/10fold-list"
out_dir <- paste0("/home_song/ybchai/nam/module/bayesA/", trait)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =============================
# 1. 读取基因型（关键修复点）
# =============================
geno <- fread(geno_file, data.table = FALSE)

ids <- geno$IID

X <- geno[, !(colnames(geno) %in%
  c("FID","IID","PAT","MAT","SEX","PHENOTYPE"))]

# ✅ 强制 numeric（核心）
X <- as.matrix(X)
storage.mode(X) <- "double"

# 把 NA 强制转为 0（BayesA 必须）
X[is.na(X)] <- 0

# 标准化
X <- scale(X, center = TRUE, scale = TRUE)

# 再次保险
storage.mode(X) <- "double"

cat("Genotype matrix:", dim(X), "\n")

# =============================
# 2. 表型
# =============================
pheno <- fread(pheno_file, header = FALSE)
setnames(pheno, c("ID","BLUP"))

pheno <- pheno[match(ids, ID)]
y <- as.numeric(pheno$BLUP)

# =============================
# 3. 十折 CV
# =============================
all_obs <- c()
all_pred <- c()

for (fold in 1:10) {

  cat("Fold", fold, "\n")

  test_ids <- fread(
    paste0(fold_dir, "/group", fold, "_ids-FIID.txt"),
    header = FALSE
  )[[1]]

  test_idx <- which(ids %in% test_ids)

  y_train <- y
  y_train[test_idx] <- NA

  ETA <- list(
    list(X = X, model = "BayesA")
  )

  fit <- BGLR(
    y = y_train,
    ETA = ETA,
    nIter = 12000,
    burnIn = 4000,
    verbose = FALSE
  )

  y_hat <- fit$yHat[test_idx]

  fwrite(
    data.frame(
      ID = ids[test_idx],
      Observed = y[test_idx],
      Predicted = y_hat,
      Fold = fold
    ),
    file = paste0(out_dir, "/fold", fold, "_prediction.txt"),
    sep = "\t"
  )

  r_fold <- cor(y[test_idx], y_hat)
  cat("  r =", round(r_fold, 4), "\n")

  all_obs <- c(all_obs, y[test_idx])
  all_pred <- c(all_pred, y_hat)
}

# =============================
# 4. 汇总
# =============================
r_all <- cor(all_obs, all_pred)

writeLines(
  c(
    paste("Trait:", trait),
    paste("Total samples:", length(all_obs)),
    paste("10-fold BayesA correlation:", round(r_all, 4))
  ),
  con = paste0(out_dir, "/bayesA_cv_summary.txt")
)

cat("Finished BayesA:", trait, "r =", round(r_all, 4), "\n")
