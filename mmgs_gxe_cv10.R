#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(MMGS)
})

read_args <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    a <- argv[[i]]
    if (a %in% c("--out-root", "--out_dir", "--out")) {
      out$out_root <- argv[[i + 1]]
      i <- i + 2
    } else if (a %in% c("--traits", "--trait")) {
      out$traits <- strsplit(argv[[i + 1]], ",", fixed = TRUE)[[1]]
      i <- i + 2
    } else if (a %in% c("--folds", "--fold")) {
      out$folds <- as.integer(strsplit(argv[[i + 1]], ",", fixed = TRUE)[[1]])
      i <- i + 2
    } else if (a %in% c("--run-id", "--runid")) {
      out$run_id <- argv[[i + 1]]
      i <- i + 2
    } else if (a %in% c("--resume-dir", "--resume_dir")) {
      out$resume_dir <- argv[[i + 1]]
      i <- i + 2
    } else if (a %in% c("--geno-file", "--geno_file")) {
      out$geno_file <- argv[[i + 1]]
      i <- i + 2
    } else if (a %in% c("--max-d1", "--max_d1")) {
      out$max_d1 <- as.numeric(argv[[i + 1]])
      i <- i + 2
    } else if (a %in% c("--max-d2", "--max_d2")) {
      out$max_d2 <- as.numeric(argv[[i + 1]])
      i <- i + 2
    } else if (a %in% c("--para-candidates")) {
      out$para_candidates <- strsplit(argv[[i + 1]], ",", fixed = TRUE)[[1]]
      i <- i + 2
    } else if (a %in% c("--mmgp-fold")) {
      out$mmgp_fold <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else if (a %in% c("--mmgp-reshuffle")) {
      out$mmgp_reshuffle <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else {
      stop("Unknown argument: ", a)
    }
  }
  out
}

safe_cor <- function(obs, pre) {
  obs <- as.numeric(obs)
  pre <- as.numeric(pre)
  ok <- is.finite(obs) & is.finite(pre)
  if (sum(ok) < 3) return(NA_real_)
  cor(obs[ok], pre[ok], use = "complete.obs")
}

estimate_env_mean_cor <- function(pheno_train, pheno_test) {
  env_cols <- setdiff(colnames(pheno_train), "line_code")
  if (length(env_cols) < 1) return(NA_real_)

  mu <- sapply(env_cols, function(ec) mean(as.numeric(pheno_train[[ec]]), na.rm = TRUE))
  pred_mat <- as.data.frame(matrix(NA_real_, nrow = nrow(pheno_test), ncol = length(env_cols)))
  colnames(pred_mat) <- env_cols
  for (ec in env_cols) pred_mat[[ec]] <- mu[[ec]]

  obs <- as.numeric(as.matrix(pheno_test[, env_cols, drop = FALSE]))
  pre <- as.numeric(as.matrix(pred_mat))
  safe_cor(obs, pre)
}

# MMGS::EPM() 在单次 Paras 个数 >= 5 时会报错：undefined columns selected（内部矩阵列索引越界）。
# 单次仅 1 个 Para 也会报错。故将 Paras 拆成多批（每批至多 4 且至少 2），再按 env_code 合并列。
epm_split_paras <- function(paras, max_per_call = 4L, min_per_call = 2L) {
  n <- length(paras)
  if (n < min_per_call) {
    stop("EPM() requires at least ", min_per_call, " Paras; got ", n)
  }
  if (n <= max_per_call) return(list(paras))
  out <- list()
  i <- 1L
  while (i <= n) {
    rem <- n - i + 1L
    if (rem <= max_per_call) {
      out[[length(out) + 1L]] <- paras[seq.int(i, n)]
      break
    }
    take <- as.integer(max_per_call)
    rem_after <- rem - take
    if (rem_after == 1L) take <- take - 1L
    out[[length(out) + 1L]] <- paras[seq.int(i, i + take - 1L)]
    i <- i + take
  }
  out
}

epm_merge_batches <- function(env_trait, env_mmgs, max_d1, max_d2, epm_paras) {
  chunks <- epm_split_paras(epm_paras)
  if (length(chunks) > 1L) {
    cat("EPM: splitting ", length(epm_paras), " Paras into ", length(chunks), " calls (MMGS EPM column limit).\n")
  }
  out <- NULL
  for (ch in chunks) {
    part <- EPM(
      data = env_trait,
      env_paras = env_mmgs,
      max_d1 = max_d1,
      max_d2 = max_d2,
      Paras = ch
    )
    pc <- ch[ch %in% colnames(part)]
    if (is.null(out)) {
      out <- part
    } else {
      part2 <- part[, c("env_code", pc), drop = FALSE]
      out <- merge(as.data.frame(out), part2, by = "env_code", all.x = TRUE, sort = FALSE)
    }
  }
  out
}

main <- function() {
  opt <- read_args()

  out_root <- opt$out_root %||% "/home_song/ybchai/nam/mmgs/results"
  run_id <- opt$run_id %||% format(Sys.time(), "%Y%m%d_%H%M%S")
  res_dir <- opt$resume_dir %||% file.path(out_root, paste0("mmgs_gxe_cv10_", run_id))
  dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

  # Redirect logs into the results directory
  sink(file.path(res_dir, "stdout_stderr.log"), split = TRUE)
  on.exit(sink(), add = TRUE)

  cat("run_id:", run_id, "\n")
  cat("res_dir:", res_dir, "\n")
  cat("start_time:", as.character(Sys.time()), "\n")
  cat("R_version:", R.version.string, "\n")

  trait_map <- list(
    spr = list(
      trait = "/home_song/ybchai/nam/mmgs/trait_flower_spr_sorted.rda",
      env_mmgs = "/home_song/ybchai/nam/mmgs/flower_spr-env_mmgs.rda"
    ),
    win = list(
      trait = "/home_song/ybchai/nam/mmgs/trait_flower_win_sorted.rda",
      env_mmgs = "/home_song/ybchai/nam/mmgs/flower_win-env_mmgs.rda"
    ),
    weight = list(
      trait = "/home_song/ybchai/nam/mmgs/trait_weight_sorted.rda",
      env_mmgs = "/home_song/ybchai/nam/mmgs/weight-env_mmgs.rda"
    ),
    oil = list(
      trait = "/home_song/ybchai/nam/mmgs/trait_oil_sorted.rda",
      env_mmgs = "/home_song/ybchai/nam/mmgs/oil-env_mmgs.rda"
    )
  )

  traits <- opt$traits %||% names(trait_map)
  folds <- opt$folds %||% 1:10
  folds <- sort(unique(folds))

  max_d1 <- opt$max_d1 %||% 18
  max_d2 <- opt$max_d2 %||% 43
  # 与 env_mmgs 中列名一致：遍历这些 Para_Name，逐个性状选十折均值最高者
  default_para_candidates <- c("FGDD", "Ftemp", "Fday", "Frad", "Fradcum", "Fprecip")
  para_candidates <- opt$para_candidates %||% default_para_candidates
  mmgp_fold <- opt$mmgp_fold %||% 2L
  mmgp_reshuffle <- opt$mmgp_reshuffle %||% 2L

  geno_file <- opt$geno_file %||% "/home_song/ybchai/nam/mmgs/geno_mmgs.rda"
  env_file <- "/home_song/ybchai/nam/mmgs/env_info.rda"
  fold_dir <- "/home_song/ybchai/nam/10fold-list"

  # Load genotype in isolated environment; avoid accidental reuse of stale objects.
  cat("geno_file:", geno_file, "\n")
  geno_env <- new.env(parent = emptyenv())
  loaded_objs <- load(geno_file, envir = geno_env)
  cat("geno_file objects:", paste(loaded_objs, collapse = ","), "\n")
  if ("geno_mmgs_fixed" %in% loaded_objs) {
    geno <- get("geno_mmgs_fixed", envir = geno_env, inherits = FALSE)
    picked <- "geno_mmgs_fixed"
  } else if ("geno_mmgs" %in% loaded_objs) {
    geno <- get("geno_mmgs", envir = geno_env, inherits = FALSE)
    picked <- "geno_mmgs"
  } else if ("geno" %in% loaded_objs) {
    geno <- get("geno", envir = geno_env, inherits = FALSE)
    picked <- "geno"
  } else {
    picked <- NULL
    for (nm in loaded_objs) {
      x <- get(nm, envir = geno_env, inherits = FALSE)
      if ((is.data.frame(x) || is.matrix(x)) && "line_code" %in% colnames(x)) {
        picked <- nm
        geno <- x
        break
      }
    }
    if (is.null(picked)) {
      stop("No genotype object with 'line_code' found in ", geno_file, ".")
    }
  }
  cat("Using genotype object:", picked, "\n")
  geno$line_code <- as.character(geno$line_code)
  # Fast monomorphism diagnostic (sampled markers)
  marker_cols <- setdiff(colnames(geno), "line_code")
  if (length(marker_cols) < 2L) {
    stop("Genotype has too few marker columns in ", geno_file)
  }
  n_check <- min(3000L, length(marker_cols))
  set.seed(20260324)
  chk_idx <- sample(marker_cols, n_check)
  poly_count <- 0L
  for (mc in chk_idx) {
    vals <- unique(geno[[mc]][!is.na(geno[[mc]])])
    if (length(vals) >= 2L) poly_count <- poly_count + 1L
  }
  geno_is_monomorphic <- (poly_count == 0L)
  cat("geno polymorphic markers in sample:", poly_count, "/", n_check, "\n")
  if (geno_is_monomorphic) {
    stop(
      "Genotype appears monomorphic in sampled markers (0/", n_check, "). ",
      "Please check --geno-file and ensure it is the rebuilt dosage matrix (e.g. geno_mmgs_fixed.rda)."
    )
  }

  load(env_file) # creates env_info
  env_info <- env_info
  env_info$env_code <- as.character(env_info$env_code)

  # Load folds ids
  fold_ids <- vector("list", 10)
  for (k in 1:10) {
    fp <- file.path(fold_dir, paste0("group", k, "_ids-FIID.txt"))
    if (!file.exists(fp)) stop("Fold id file not found: ", fp)
    ids <- fread(fp, header = FALSE)[[1]]
    fold_ids[[k]] <- as.character(ids)
  }

  cat("traits:", paste(traits, collapse = ","), "\n")
  cat("folds:", paste(folds, collapse = ","), "\n")

  # Trait-level summary（best_Para_Name = 该性状在候选环境变量中十折均值最高者）
  trait_summary <- data.table(
    trait = traits,
    best_Para_Name = NA_character_,
    mean_correlation = NA_real_,
    sd_correlation = NA_real_,
    valid_folds = 0L
  )

  for (trait_name in traits) {
    cat("\n=============================\n")
    cat("Trait:", trait_name, "\n")
    cat("=============================\n")

    tdir <- file.path(res_dir, trait_name)
    dir.create(tdir, recursive = TRUE, showWarnings = FALSE)

    # Load phenotype and environment parameter seeds
    load(trait_map[[trait_name]]$trait) # creates long
    trait_long <- long
    trait_long$line_code <- as.character(trait_long$line_code)
    trait_long$env_code <- as.character(trait_long$env_code)
    trait_long$pop_code <- as.character(trait_long$pop_code)
    trait_long$Trait <- as.numeric(trait_long$Trait)

    load(trait_map[[trait_name]]$env_mmgs) # creates env_mmgs
    env_mmgs <- env_mmgs

    # Prepare MMGS inputs
    env_trait <- env_trait_calculate(data = trait_long, trait = "Trait", env = "env_code")
    LbyE <- LbyE_calculate(data = trait_long, trait = "Trait", env = "env_code", line = "line_code")

    epm_paras <- intersect(para_candidates, colnames(env_mmgs))
    if (length(epm_paras) < 1L) {
      stop("No Para_Name from candidates exists in env_mmgs colnames for trait ", trait_name)
    }

    envMeanPara <- epm_merge_batches(
      env_trait = env_trait,
      env_mmgs = env_mmgs,
      max_d1 = max_d1,
      max_d2 = max_d2,
      epm_paras = epm_paras
    )

    saveRDS(LbyE, file.path(tdir, paste0("LbyE_", trait_name, ".rds")))
    saveRDS(envMeanPara, file.path(tdir, paste0("envMeanPara_", trait_name, ".rds")))

    env_cols <- setdiff(colnames(LbyE), "line_code")
    if (length(env_cols) < 2) stop("Too few environment columns for ", trait_name)

    paras_to_run <- epm_paras[epm_paras %in% colnames(envMeanPara)]
    cat("Para_Name grid (EPM columns):", paste(paras_to_run, collapse = ","), "\n")

    all_para_fold_rows <- list()
    para_compare_rows <- list()

    for (chosen_para in paras_to_run) {
      cat("\n  --- Para_Name:", chosen_para, "---\n")

      folds_dt <- data.table(
        trait = trait_name,
        fold = folds,
        model_used = "MMGS",
        Para_Name = chosen_para,
        n_train = NA_integer_,
        n_test = NA_integer_,
        n_pred = NA_integer_,
        correlation = NA_real_,
        error = NA_character_
      )

      # Resume support: reuse completed folds (finite correlation) in the same run directory.
      folds_fp <- file.path(tdir, paste0("trait_fold_results_", trait_name, "_", chosen_para, ".csv"))
      if (file.exists(folds_fp)) {
        old_dt <- tryCatch(fread(folds_fp), error = function(e) NULL)
        if (!is.null(old_dt) && nrow(old_dt) > 0L && "fold" %in% colnames(old_dt)) {
          keep_cols <- intersect(colnames(old_dt), colnames(folds_dt))
          old_dt <- old_dt[fold %in% folds, ..keep_cols]
          if (nrow(old_dt) > 0L) {
            # 按 fold 对齐写回（避免 joins 里对 i 的依赖，兼容不同 data.table 版本）
            m <- match(folds_dt$fold, old_dt$fold)
            for (cn in setdiff(keep_cols, "fold")) {
              src <- old_dt[[cn]][m]
              set(folds_dt, j = cn, value = src)
            }
          }
        }
      }

      for (k in folds) {
        if (is.finite(folds_dt[fold == k, correlation])) {
          cat("    Fold:", k, "...skip (already done)\n")
          next
        }
        cat("    Fold:", k, "...\n")

        ids_test <- intersect(fold_ids[[k]], geno$line_code)
        ids_train <- setdiff(unique(LbyE$line_code), ids_test)

        pheno_train <- LbyE[LbyE$line_code %in% ids_train, ]
        pheno_test <- LbyE[LbyE$line_code %in% ids_test, ]

        pheno_train <- pheno_train[complete.cases(pheno_train[, env_cols, drop = FALSE]), ]
        pheno_test <- pheno_test[complete.cases(pheno_test[, env_cols, drop = FALSE]), ]

        folds_dt[fold == k, n_train := nrow(pheno_train)]
        folds_dt[fold == k, n_test := nrow(pheno_test)]

        if (nrow(pheno_train) < 20 || nrow(pheno_test) < 5) {
          folds_dt[fold == k, error := "Too few rows after filtering/complete-cases."]
          next
        }

        fit <- tryCatch({
          MMGP(
            pheno = as.data.frame(pheno_train),
            geno = geno,
            env = env_info,
            para = envMeanPara,
            Para_Name = chosen_para,
            model = "rrBLUP",
            depend = "Norm",
            fold = mmgp_fold,
            reshuffle = mmgp_reshuffle,
            methods = "RM.G",
            ms1 = 2,
            ms2 = 2
          )
        }, error = function(e) e)

        if (inherits(fit, "error")) {
          folds_dt[fold == k, error := fit$message]
          next
        }

        prd <- tryCatch({
          MMPrdM(
            pheno = as.data.frame(pheno_test),
            geno = geno,
            env = env_info,
            para = envMeanPara,
            model = "rrBLUPJ",
            depend = "PEI",
            Para_Name = chosen_para,
            reshuffle = 2
          )
        }, error = function(e) e)

        if (inherits(prd, "error")) {
          folds_dt[fold == k, error := prd$message]
          next
        }

        pred_obj <- prd[[1]]
        obs <- pred_obj$obs
        pre <- pred_obj$pre

        cor_k <- safe_cor(obs, pre)
        folds_dt[fold == k, n_pred := length(obs)]
        folds_dt[fold == k, correlation := cor_k]

        pred_tag <- paste0("pred_", trait_name, "_", chosen_para, "_fold", k)
        saveRDS(pred_obj, file.path(tdir, paste0(pred_tag, ".rds")))
        pred_df <- as.data.frame(pred_obj)
        fwrite(pred_df, file.path(tdir, paste0(pred_tag, ".tsv")), sep = "\t")
      }

      fwrite(folds_dt, file.path(tdir, paste0("trait_fold_results_", trait_name, "_", chosen_para, ".csv")))

      valid <- folds_dt[!is.na(correlation) & is.finite(correlation)]
      mean_cor <- if (nrow(valid) > 0) mean(valid$correlation) else NA_real_
      sd_cor <- if (nrow(valid) > 1) sd(valid$correlation) else NA_real_

      para_compare_rows[[length(para_compare_rows) + 1L]] <- data.table(
        trait = trait_name,
        Para_Name = chosen_para,
        mean_correlation = mean_cor,
        sd_correlation = sd_cor,
        valid_folds = as.integer(nrow(valid))
      )

      all_para_fold_rows[[length(all_para_fold_rows) + 1L]] <- folds_dt

      cat("    Para_Name", chosen_para, "valid_folds:", nrow(valid), "mean r:", mean_cor, "\n")
    }

    para_compare_dt <- rbindlist(para_compare_rows)
    fwrite(para_compare_dt, file.path(tdir, paste0("para_compare_summary_", trait_name, ".csv")))

    if (nrow(para_compare_dt) > 0L) {
      mc <- as.numeric(para_compare_dt$mean_correlation)
      okv <- is.finite(mc)
      w <- if (any(okv)) which.max(replace(mc, !okv, -Inf)) else NA_integer_
      if (!is.na(w) && length(w) == 1L && is.finite(mc[[w]])) {
        trait_summary[trait == trait_name, best_Para_Name := para_compare_dt$Para_Name[[w]]]
        trait_summary[trait == trait_name, mean_correlation := para_compare_dt$mean_correlation[[w]]]
        trait_summary[trait == trait_name, sd_correlation := para_compare_dt$sd_correlation[[w]]]
        trait_summary[trait == trait_name, valid_folds := para_compare_dt$valid_folds[[w]]]
        cat(
          "BEST for ", trait_name, ": ", para_compare_dt$Para_Name[[w]],
          " mean=", para_compare_dt$mean_correlation[[w]], "\n",
          sep = ""
        )
      }
    }

    if (length(all_para_fold_rows) > 0L) {
      fwrite(rbindlist(all_para_fold_rows), file.path(tdir, paste0("all_para_fold_results_", trait_name, ".csv")))
    }
  }

  if (length(traits) > 0L) {
    parts <- lapply(traits, function(tn) {
      fp <- file.path(res_dir, tn, paste0("para_compare_summary_", tn, ".csv"))
      if (file.exists(fp)) fread(fp) else NULL
    })
    parts <- Filter(Negate(is.null), parts)
    if (length(parts) > 0L) {
      combined <- rbindlist(parts, use.names = TRUE, fill = TRUE)
      if (nrow(combined) > 0L) {
        fwrite(combined, file.path(res_dir, "mmgs_para_compare_all_traits.csv"))
      }
    }
  }

  summary_fp <- file.path(res_dir, "mmgs_summary_all_traits.csv")
  fwrite(trait_summary, summary_fp)

  cat("\nFinal summary saved to:\n", summary_fp, "\n")
  cat("end_time:", as.character(Sys.time()), "\n")
}

`%||%` <- function(a, b) {
  if (!is.null(a)) a else b
}

main()

