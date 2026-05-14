#!/usr/bin/env Rscript

# 安装BGLR（如果未安装）
if(!require(BGLR)) {
  install.packages("BGLR")
  library(BGLR)
}

library(data.table)
library(Matrix)

# -----------------------------
# 参数设置
# -----------------------------
grm_prefix <- "/home_song/ybchai/nam/model/NAM_gblup"
pheno_dir  <- "/home_song/ybchai/nam/pheno_BLUP"
out_dir    <- "/home_song/ybchai/nam/model/gblup-bglr"
traits     <- c("flower_win", "flower_spr", "oil", "weight")
nfolds     <- 10

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# 读取 GRM
# -----------------------------
read_gcta_grm <- function(prefix) {
  id  <- fread(paste0(prefix, ".grm.id"), header = FALSE)
  ids <- id[[2]]
  n   <- length(ids)
  con <- file(paste0(prefix, ".grm.bin"), "rb")
  grm_values <- readBin(con, what = "numeric", size = 4, n = n * (n + 1) / 2)
  close(con)
  
  K <- matrix(0, n, n)
  K[lower.tri(K, diag = TRUE)] <- grm_values
  K <- K + t(K) - diag(diag(K))
  rownames(K) <- colnames(K) <- ids
  return(K)
}

cat("Reading GRM ...\n")
K <- read_gcta_grm(grm_prefix)
cat("GRM loaded. Dimension:", dim(K)[1], dim(K)[2], "\n\n")

# -----------------------------
# 十折交叉验证
# -----------------------------
for(trait in traits){
  cat("Processing trait:", trait, "\n")
  
  pheno_file <- file.path(pheno_dir, paste0(trait, "_BLUP.txt"))
  if(!file.exists(pheno_file)){
    cat("  Error: Phenotype file not found. Skipping.\n")
    next
  }
  
  pheno <- fread(pheno_file, data.table = FALSE)
  colnames(pheno)[1:2] <- c("ID", "BLUP")
  
  # 对齐数据
  ids_in_both <- intersect(pheno$ID, rownames(K))
  pheno <- pheno[pheno$ID %in% ids_in_both, ]
  K_sub <- K[ids_in_both, ids_in_both]
  
  # 确保顺序一致
  pheno <- pheno[match(rownames(K_sub), pheno$ID), ]
  
  cat("  Number of samples:", nrow(pheno), "\n")
  cat("  Phenotype mean:", mean(pheno$BLUP), "\n")
  cat("  Phenotype range:", range(pheno$BLUP), "\n")
  
  # 创建10折
  set.seed(123)
  folds <- sample(rep(1:nfolds, length.out = nrow(pheno)))
  
  # 存储预测结果
  predictions <- data.frame(
    ID = pheno$ID, 
    Observed = pheno$BLUP, 
    Predicted = NA,
    Fold = folds
  )
  
  for(f in 1:nfolds) {
    cat("  Fold:", f, "/", nfolds, "\n")
    
    # 训练集和测试集索引
    train_idx <- which(folds != f)
    test_idx <- which(folds == f)
    
    cat("    Training set size:", length(train_idx), "\n")
    cat("    Test set size:", length(test_idx), "\n")
    
    # 准备数据
    y_train <- pheno$BLUP[train_idx]
    
    # 准备关系矩阵
    K_train <- K_sub[train_idx, train_idx]
    
    # 标准化y_train（可选，但BGLR会自动处理）
    # y_train <- scale(y_train)
    
    # BGLR模型设置
    ETA <- list(
      list(K = K_train, model = "RKHS")
    )
    
    # 拟合模型
    fm <- BGLR(
      y = y_train,
      ETA = ETA,
      nIter = 12000,
      burnIn = 2000,
      thin = 5,
      verbose = FALSE
    )
    
    # 正确的预测方法
    # 方法1：使用BGLR的预测函数
    # 对于测试集，我们需要计算：y_pred = mu + K_test_train %*% (solve(K_train) %*% u)
    
    # 提取参数
    mu <- fm$mu  # 总体均值（固定效应）
    u <- fm$ETA[[1]]$u  # 遗传效应（BLUP）
    
    cat("    Estimated mu:", mu, "\n")
    cat("    Variance of u:", var(u), "\n")
    
    # 计算训练集的预测值（用于验证）
    y_train_pred <- mu + K_train %*% u
    train_cor <- cor(y_train, y_train_pred)
    cat("    Training correlation:", train_cor, "\n")
    
    # 预测测试集
    if(length(test_idx) > 0) {
      K_test_train <- K_sub[test_idx, train_idx, drop = FALSE]
      
      # 正确预测：总体均值 + 加权遗传效应
      pred <- mu + as.vector(K_test_train %*% u)
      
      # 存储预测值
      predictions$Predicted[test_idx] <- pred
      
      # 显示前几个预测值用于检查
      cat("    Sample predictions (first 5):\n")
      for(i in 1:min(5, length(pred))) {
        cat("      ", pheno$ID[test_idx[i]], ":", 
            round(pred[i], 2), "(true:", pheno$BLUP[test_idx[i]], ")\n")
      }
    }
  }
  
  # 计算预测精度
  valid_idx <- !is.na(predictions$Predicted)
  cat("\n  Final Results for", trait, ":\n")
  cat("    Number of predictions:", sum(valid_idx), "/", nrow(predictions), "\n")
  
  if(sum(valid_idx) > 10) {
    obs <- predictions$Observed[valid_idx]
    pred <- predictions$Predicted[valid_idx]
    
    # 计算各种统计量
    cor_val <- cor(obs, pred)
    mse_val <- mean((obs - pred)^2)
    bias <- mean(pred - obs)
    
    cat("    Correlation (r):", round(cor_val, 4), "\n")
    cat("    MSE:", round(mse_val, 4), "\n")
    cat("    Mean bias:", round(bias, 4), "\n")
    cat("    Observed range:", round(range(obs), 2), "\n")
    cat("    Predicted range:", round(range(pred), 2), "\n")
    
    # 绘制预测vs观测图
    png(file.path(out_dir, paste0(trait, "_pred_vs_obs.png")))
    plot(obs, pred, 
         xlab = "Observed", ylab = "Predicted",
         main = paste(trait, "- Correlation:", round(cor_val, 3)))
    abline(0, 1, col = "red")
    abline(lm(pred ~ obs), col = "blue", lty = 2)
    legend("topleft", legend = c("1:1 line", "Regression line"), 
           col = c("red", "blue"), lty = c(1, 2))
    dev.off()
  }
  
  # 保存结果
  out_file <- file.path(out_dir, paste0(trait, "_GBLUP_pred_corrected.txt"))
  write.table(predictions, file = out_file, sep = "\t", 
              row.names = FALSE, quote = FALSE)
  cat("  Saved to:", out_file, "\n\n")
}