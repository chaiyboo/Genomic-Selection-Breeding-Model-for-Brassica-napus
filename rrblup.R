#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(rrBLUP)
})

normalize_id <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- tolower(x)
  x <- gsub("[[:space:]]+", "", x)
  x <- gsub("[-_.]", "", x)
  x
}

to_numeric_matrix <- function(df) {
  x <- as.data.frame(df, stringsAsFactors = FALSE)
  for (j in seq_len(ncol(x))) {
    x[[j]] <- suppressWarnings(as.numeric(as.character(x[[j]])))
  }
  keep <- colSums(!is.na(x)) > 0
  x <- x[, keep, drop = FALSE]
  m <- as.matrix(x)
  for (j in seq_len(ncol(m))) {
    idx <- is.na(m[, j])
    if (any(idx)) {
      mu <- mean(m[, j], na.rm = TRUE)
      if (is.na(mu)) mu <- 0
      m[idx, j] <- mu
    }
  }
  m
}

select_best_geno_id <- function(geno_dt, pheno_ids) {
  pheno_key <- unique(normalize_id(pheno_ids))
  candidates <- list()

  rn <- rownames(geno_dt)
  if (!is.null(rn) && length(rn) == nrow(geno_dt)) {
    candidates[[".rownames"]] <- rn
  }
  for (cn in colnames(geno_dt)) {
    candidates[[cn]] <- geno_dt[[cn]]
  }

  best_name <- NULL
  best_overlap <- -1L
  for (nm in names(candidates)) {
    vec <- candidates[[nm]]
    key <- normalize_id(vec)
    overlap <- sum(unique(key) %in% pheno_key, na.rm = TRUE)
    if (overlap > best_overlap) {
      best_overlap <- overlap
      best_name <- nm
    }
  }
  list(name = best_name, overlap = best_overlap)
}

run_cv_rrblup <- function(y, M, fold_n, seed) {
  n <- length(y)
  set.seed(seed)
  fold_id <- sample(rep(seq_len(fold_n), length.out = n))
  pred <- rep(NA_real_, n)

  for (k in seq_len(fold_n)) {
    test_idx <- which(fold_id == k)
    y_train <- y
    y_train[test_idx] <- NA_real_

    fit <- rrBLUP::mixed.solve(
      y = y_train,
      Z = M,
      X = matrix(1, nrow = n, ncol = 1)
    )
    beta0 <- if (length(fit$beta) > 0) as.numeric(fit$beta)[1] else 0
    g_hat <- as.vector(M %*% fit$u)
    pred_all <- beta0 + g_hat
    pred[test_idx] <- pred_all[test_idx]
  }
  list(pred = pred, fold_id = fold_id)
}

out_dir <- "/home_song/ybchai/nam/model/rrblup-mmgs/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
intermediate_dir <- file.path(out_dir, "intermediate")
result_dir <- file.path(out_dir, "results")
dir.create(intermediate_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

CV_FOLD <- as.integer(Sys.getenv("CV_FOLD", "10"))
CV_SEED <- as.integer(Sys.getenv("CV_SEED", "20260421"))
if (is.na(CV_FOLD) || CV_FOLD < 2) stop("CV_FOLD 必须 >= 2")

cat("CV_FOLD =", CV_FOLD, "\n")
cat("CV_SEED =", CV_SEED, "\n")
cat("模型设定: rrBLUP 无环境协变量 (仅截距 + 基因型)\n")

geno_file <- "/home_song/ybchai/nam/mmgs/geno_mmgs_fixed.rda"
geno_loaded <- load(geno_file)
if (length(geno_loaded) == 0) stop("geno 文件未加载到对象")
geno_raw <- get(geno_loaded[1])
geno_dt <- as.data.table(geno_raw)

trait_files <- list(
  spr    = "/home_song/ybchai/nam/mmgs/trait_flower_spr_sorted.rda",
  win    = "/home_song/ybchai/nam/mmgs/trait_flower_win_sorted.rda",
  weight = "/home_song/ybchai/nam/mmgs/trait_weight_sorted.rda",
  oil    = "/home_song/ybchai/nam/mmgs/trait_oil_sorted.rda"
)

status <- data.table(
  trait = character(),
  n_raw = integer(),
  n_agg = integer(),
  n_common = integer(),
  id_source = character(),
  model_status = character(),
  note = character()
)
all_pred <- data.table()

for (trait_name in names(trait_files)) {
  cat("开始性状:", trait_name, "\n")
  trait_intermediate <- file.path(intermediate_dir, trait_name)
  dir.create(trait_intermediate, showWarnings = FALSE, recursive = TRUE)

  objn <- load(trait_files[[trait_name]])
  if (length(objn) == 0) {
    status <- rbind(status, data.table(
      trait = trait_name, n_raw = 0L, n_agg = 0L, n_common = 0L,
      id_source = NA_character_, model_status = "skip", note = "表型RData为空"
    ))
    next
  }
  pheno <- as.data.table(get(objn[1]))
  if (!all(c("line_code", "Trait") %in% colnames(pheno))) {
    status <- rbind(status, data.table(
      trait = trait_name, n_raw = nrow(pheno), n_agg = 0L, n_common = 0L,
      id_source = NA_character_, model_status = "skip", note = "缺少 line_code/Trait"
    ))
    next
  }

  pheno <- pheno[!is.na(line_code) & !is.na(Trait)]
  pheno[, line_code := as.character(line_code)]
  pheno[, Trait := suppressWarnings(as.numeric(Trait))]
  pheno <- pheno[is.finite(Trait)]
  n_raw <- nrow(pheno)
  pheno_agg <- pheno[, .(Trait = mean(as.numeric(Trait), na.rm = TRUE)), by = line_code]
  pheno_agg <- pheno_agg[!is.na(Trait)]
  n_agg <- nrow(pheno_agg)
  fwrite(pheno_agg, file.path(trait_intermediate, "pheno_no_env.tsv"), sep = "\t")

  id_pick <- select_best_geno_id(geno_dt, pheno_agg$line_code)
  id_source <- id_pick$name
  if (is.null(id_source)) {
    status <- rbind(status, data.table(
      trait = trait_name, n_raw = n_raw, n_agg = n_agg, n_common = 0L,
      id_source = NA_character_, model_status = "skip", note = "未找到可用基因型ID来源"
    ))
    next
  }

  if (id_source == ".rownames") {
    geno_ids <- rownames(geno_dt)
    geno_marker_dt <- copy(geno_dt)
  } else {
    geno_ids <- geno_dt[[id_source]]
    geno_marker_dt <- copy(geno_dt)
    geno_marker_dt[[id_source]] <- NULL
  }
  geno_ids <- as.character(geno_ids)

  geno_map <- data.table(geno_id = geno_ids, geno_key = normalize_id(geno_ids))
  geno_map <- unique(geno_map, by = "geno_key")
  pheno_map <- copy(pheno_agg)
  pheno_map[, pheno_key := normalize_id(line_code)]
  pheno_map <- unique(pheno_map, by = "pheno_key")

  map_dt <- merge(
    pheno_map[, .(line_code, Trait, pheno_key)],
    geno_map[, .(geno_id, geno_key)],
    by.x = "pheno_key",
    by.y = "geno_key",
    all = FALSE
  )
  n_common <- nrow(map_dt)
  fwrite(map_dt, file.path(trait_intermediate, "line_code_map.tsv"), sep = "\t")

  if (n_common < CV_FOLD) {
    status <- rbind(status, data.table(
      trait = trait_name, n_raw = n_raw, n_agg = n_agg, n_common = n_common,
      id_source = id_source, model_status = "skip",
      note = paste0("重叠样本数(", n_common, ")小于折数(", CV_FOLD, ")")
    ))
    next
  }

  row_idx <- match(map_dt$geno_id, geno_ids)
  geno_use <- geno_marker_dt[row_idx, , drop = FALSE]
  M <- to_numeric_matrix(geno_use)
  if (ncol(M) == 0) {
    status <- rbind(status, data.table(
      trait = trait_name, n_raw = n_raw, n_agg = n_agg, n_common = n_common,
      id_source = id_source, model_status = "skip", note = "基因型标记列在数值化后为空"
    ))
    next
  }
  rownames(M) <- map_dt$line_code
  saveRDS(M, file.path(trait_intermediate, "geno_matrix_used.rds"))

  cv <- run_cv_rrblup(
    y = map_dt$Trait,
    M = M,
    fold_n = CV_FOLD,
    seed = CV_SEED
  )

  pred_dt <- data.table(
    trait = trait_name,
    line_code = map_dt$line_code,
    fold = cv$fold_id,
    observed = map_dt$Trait,
    predicted = cv$pred
  )
  pred_dt[, residual := observed - predicted]
  fwrite(pred_dt, file.path(result_dir, paste0(trait_name, "_cv", CV_FOLD, "_predictions.tsv")), sep = "\t")
  all_pred <- rbind(all_pred, pred_dt, fill = TRUE)

  metric_dt <- pred_dt[, .(
    n = .N,
    cor = suppressWarnings(cor(observed, predicted, use = "complete.obs")),
    rmse = sqrt(mean((observed - predicted)^2, na.rm = TRUE))
  ), by = .(trait, fold)]
  fwrite(metric_dt, file.path(result_dir, paste0(trait_name, "_cv", CV_FOLD, "_metrics.tsv")), sep = "\t")

  status <- rbind(status, data.table(
    trait = trait_name, n_raw = n_raw, n_agg = n_agg, n_common = n_common,
    id_source = id_source, model_status = "ok", note = paste0("rrBLUP ", CV_FOLD, "折完成(无环境协变量)")
  ))
}

if (nrow(all_pred) > 0) {
  fwrite(all_pred, file.path(result_dir, paste0("rrblup_cv", CV_FOLD, "_all_traits_predictions.tsv")), sep = "\t")
}
fwrite(status, file.path(out_dir, "run_status.tsv"), sep = "\t")
cat("完成，状态文件:", file.path(out_dir, "run_status.tsv"), "\n")