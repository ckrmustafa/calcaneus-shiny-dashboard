# ============================================================================
# app.R
# ----------------------------------------------------------------------------
# Calcaneus Morphometry — Machine Learning Dashboard
# A publication-grade R/Shiny application for clinical classification of
# calcaneal (heel bone) morphometric findings.
#
# HIGHLIGHTS
#   * Automatic standardization of Turkish/English labels (VAR/YOK, YES/NO,
#     K/E, F/M) with typo correction.
#   * 10 classifiers under one caret pipeline: logistic regression, random
#     forest, XGBoost, GBM, SVM (RBF), k-NN, CART, Naive Bayes, neural
#     network (NNT) and LDA.
#   * 5-fold cross-validation with stratification.
#   * K-NN imputation INSIDE the resampling pipeline (no data leakage).
#   * Class-imbalance handling: SMOTE / ROSE / up- / down-sampling.
#   * Train vs. test performance: Accuracy, F1, Sensitivity, Specificity,
#     ROC-AUC; metrics table downloadable as CSV.
#   * Publication-quality graphics exportable at 600 DPI: ROC curves,
#     train/test comparison bars, feature importance, white-box decision tree.
# ============================================================================

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c(
  "shiny", "shinydashboard", "DT",               # UI
  "caret", "pROC", "dplyr", "tidyr", "ggplot2",  # ML + data + plots
  "randomForest", "xgboost", "gbm", "e1071",     # model backends
  "rpart", "rpart.plot", "naivebayes", "nnet",
  "MASS", "readxl", "readr", "stringr",
  "themis", "ROSE"                                # imbalance handling
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                      logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  message("The following packages are required. Install them with:\n",
          'install.packages(c("', paste(missing_pkgs, collapse = '", "'), '"))')
}

suppressPackageStartupMessages({
  library(shiny);        library(shinydashboard); library(DT)
  library(caret);        library(pROC);           library(dplyr)
  library(tidyr);        library(ggplot2)
  library(randomForest); library(xgboost);        library(gbm)
  library(e1071);        library(rpart);          library(rpart.plot)
  library(naivebayes);   library(nnet);           library(MASS)
  library(readxl);       library(readr);          library(stringr)
library(tidyr)   # pivot_longer for the correlation module
})

# ============================================================================
# 1. Label-standardization helpers (same logic as data_prep.R, so the app is
#    self-contained and can also process freshly uploaded raw files)
# ============================================================================
.norm_token <- function(x) {
  x <- stringr::str_trim(stringr::str_squish(as.character(x)))
  toupper(x)
}

standardize_binary <- function(x) {
  t <- .norm_token(x)
  yes_tokens <- c("VAR","V","YES","Y","EVET","PRESENT","POS","POSITIVE","1","TRUE","T")
  no_tokens  <- c("YOK","N","NO","HAYIR","H","ABSENT","NEG","NEGATIVE","0","FALSE","NONE")
  out <- dplyr::case_when(
    t %in% yes_tokens           ~ "Yes",
    t %in% no_tokens            ~ "No",
    t %in% c("TOK")             ~ "No",    # observed typo in raw export
    t %in% c("VARR","VAE")      ~ "Yes",
    t == "" | t == "NA"         ~ NA_character_,
    TRUE                        ~ NA_character_
  )
  factor(out, levels = c("No", "Yes"))
}

standardize_sex <- function(x) {
  t <- .norm_token(x)
  out <- dplyr::case_when(
    t %in% c("K","F","KADIN","FEMALE","WOMAN") ~ "Female",
    t %in% c("E","M","ERKEK","MALE","MAN","EK") ~ "Male",
    t == "" | t == "NA" ~ NA_character_,
    TRUE ~ NA_character_
  )
  factor(out, levels = c("Female", "Male"))
}

safe_numeric <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  suppressWarnings(as.numeric(gsub(",", ".", .norm_token(x))))
}

# Detect column roles from a raw/clean data frame (project naming convention)
clean_dataset <- function(df) {
  names(df) <- tolower(gsub("[^A-Za-z0-9]+", "_", names(df)))
  names(df) <- gsub("_$", "", names(df))
  for (v in names(df)) {
    if (v == "sex") {
      df[[v]] <- standardize_sex(df[[v]])
    } else if (v == "age" || grepl("^(ga|ba)_", v)) {
      df[[v]] <- safe_numeric(df[[v]])
    } else if (grepl("^(hd|hs|at)_", v)) {
      df[[v]] <- standardize_binary(df[[v]])
    }
  }
  # physiological plausibility bounds -> NA (imputed later, in-pipeline)
  if ("age" %in% names(df)) df$age <- ifelse(df$age < 0 | df$age > 110, NA, df$age)
  for (v in grep("^ga_", names(df), value = TRUE))
    df[[v]] <- ifelse(df[[v]] < 60 | df[[v]] > 170, NA, df[[v]])
  for (v in grep("^ba_", names(df), value = TRUE))
    df[[v]] <- ifelse(df[[v]] < 5 | df[[v]] > 70, NA, df[[v]])
  if ("sex" %in% names(df)) df <- df[!is.na(df$sex), , drop = FALSE]
  df
}

# ============================================================================
# 2. Model registry — 10 algorithms
# ============================================================================
MODEL_REGISTRY <- list(
  "Logistic Regression" = "glm",
  "Random Forest"       = "rf",
  "XGBoost"             = "xgbDirect",   # direct xgboost engine, see below
  "GBM"                 = "gbm",
  "SVM (RBF)"           = "svmRadial",
  "k-NN"                = "knn",
  "CART"                = "rpart",
  "Naive Bayes"         = "naive_bayes",
  "Neural Network"      = "nnet",
  "LDA"                 = "lda"
)

# short labels for compact multi-panel figure legends (batch module)
MODEL_SHORT <- c("Logistic Regression" = "LR",  "Random Forest"  = "RF",
                 "XGBoost" = "XGBoost", "GBM" = "GBM", "SVM (RBF)" = "SVM-RBF",
                 "k-NN" = "k-NN", "CART" = "CART", "Naive Bayes" = "NB",
                 "Neural Network" = "NNT", "LDA" = "LDA")

# per-model extra arguments passed through caret::train()
# NOTE: do NOT pass parameters that caret's tuneGrid already controls
# (e.g. mtry for rf, decay for nnet) — this causes "Stopping" errors.
# Anti-overfitting is handled via tuneLength and the CV design instead.
model_extra_args <- function(method) {
  switch(method,
    "glm"      = list(family = binomial()),
    "gbm"      = list(verbose = FALSE, distribution = "bernoulli"),
    "nnet"     = list(trace = FALSE, maxit = 500),
    "rf"       = list(ntree = 500),
    list()
  )
}

# ============================================================================
# Direct XGBoost engine
# ----------------------------------------------------------------------------
# caret's "xgbTree" wrapper is incompatible with xgboost >= 3.x: after fitting
# it executes `modelFit$xNames <- colnames(x)`, which fails on the new
# ALTREP-based booster class with
#   "ALTLIST classes must provide a Set_elt method [XGBAltrepPointerClass]".
# XGBoost is therefore trained directly with xgboost::xgb.train while keeping
# the SAME experimental design as the other nine algorithms: stratified
# 5-fold CV, ROC-AUC as the tuning metric, and class balancing applied inside
# each fold (SMOTE / ROSE / up / down), then a final fit on the training set.
# ============================================================================

# apply the selected balancing strategy to a training (sub)set
apply_balance <- function(X, y, method) {
  if (method == "none") return(list(X = X, y = y))
  df <- data.frame(.y = y, X, check.names = FALSE)
  out <- switch(method,
    "smote" = {
      s <- themis::smote(df, var = ".y")
      list(X = s[, -1, drop = FALSE], y = s$.y)
    },
    "rose" = {
      r <- ROSE::ROSE(.y ~ ., data = df)$data
      list(X = r[, -1, drop = FALSE], y = r$.y)
    },
    "up" = {
      u <- caret::upSample(x = X, y = y)
      list(X = u[, setdiff(names(u), "Class"), drop = FALSE], y = u$Class)
    },
    "down" = {
      d <- caret::downSample(x = X, y = y)
      list(X = d[, setdiff(names(d), "Class"), drop = FALSE], y = d$Class)
    })
  out
}

train_xgb_direct <- function(X, y, balance, tune_len = 3, seed = 2025) {
  set.seed(seed)
  folds <- caret::createFolds(y, k = 5, returnTrain = FALSE)
  grid  <- expand.grid(eta = c(0.05, 0.1, 0.3), max_depth = c(3, 6),
                       KEEP.OUT.ATTRS = FALSE)
  if (tune_len < 3) grid <- grid[grid$eta == 0.1 & grid$max_depth == 6, ,
                                 drop = FALSE]

  cv_auc <- numeric(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    fold_auc <- numeric(length(folds))
    for (f in seq_along(folds)) {
      te <- folds[[f]]; tr <- setdiff(seq_len(nrow(X)), te)
      bal <- apply_balance(X[tr, , drop = FALSE], y[tr], balance)
      dtr <- xgboost::xgb.DMatrix(as.matrix(bal$X),
                                  label = as.integer(bal$y == "Yes"))
      dte <- xgboost::xgb.DMatrix(as.matrix(X[te, , drop = FALSE]),
                                  label = as.integer(y[te] == "Yes"))
      bst <- xgboost::xgb.train(
        params  = list(objective = "binary:logistic", eval_metric = "auc",
                       eta = grid$eta[g], max_depth = grid$max_depth[g],
                       subsample = 0.8, colsample_bytree = 0.8),
        data    = dtr, nrounds = 150, verbose = 0)
      pv  <- predict(bst, dte)
      fold_auc[f] <- as.numeric(pROC::auc(pROC::roc(y[te], pv, quiet = TRUE)))
    }
    cv_auc[g] <- mean(fold_auc)
  }
  best <- grid[which.max(cv_auc), , drop = FALSE]

  # final model on the (balanced) full training partition
  bal_full <- apply_balance(X, y, balance)
  dfull <- xgboost::xgb.DMatrix(as.matrix(bal_full$X),
                                label = as.integer(bal_full$y == "Yes"))
  booster <- xgboost::xgb.train(
    params  = list(objective = "binary:logistic", eval_metric = "auc",
                   eta = best$eta, max_depth = best$max_depth,
                   subsample = 0.8, colsample_bytree = 0.8),
    data    = dfull, nrounds = 150, verbose = 0)

  structure(list(booster        = booster,
                 feature_names  = colnames(X),
                 best_tune      = best,
                 results        = data.frame(ROC = max(cv_auc)),
                 levels         = c("Yes", "No")),
            class = "xgb_direct")
}

# S3 predict method: raw -> class factor, prob -> data.frame with Yes/No
predict.xgb_direct <- function(object, newdata, type = c("raw", "prob"), ...) {
  type <- match.arg(type)
  nd  <- as.data.frame(newdata)
  dm  <- xgboost::xgb.DMatrix(
    as.matrix(nd[, object$feature_names, drop = FALSE]))
  pr  <- predict(object$booster, dm)
  if (type == "prob") return(data.frame(Yes = pr, No = 1 - pr))
  factor(ifelse(pr >= 0.5, "Yes", "No"), levels = object$levels)
}

# ============================================================================
# 3. Metric computation (train/test)
# ============================================================================
compute_metrics <- function(obs, pred_class, pred_prob, set_name, model_name) {
  obs  <- factor(obs, levels = c("Yes", "No"))
  pred_class <- factor(pred_class, levels = c("Yes", "No"))
  cm <- caret::confusionMatrix(pred_class, obs, positive = "Yes")
  roc_obj <- tryCatch(
    pROC::roc(response = obs, predictor = as.numeric(pred_prob), quiet = TRUE),
    error = function(e) NULL
  )
  auc_val <- if (is.null(roc_obj)) NA_real_ else as.numeric(pROC::auc(roc_obj))
  data.frame(
    Model       = model_name,
    Set         = set_name,
    Accuracy    = unname(cm$overall["Accuracy"]),
    F1          = unname(cm$byClass["F1"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"]),
    ROC_AUC     = auc_val,
    stringsAsFactors = FALSE
  )
}

# 600-DPI export helper for base-graphics plots
save_base_600dpi <- function(file, plot_fn, width = 8, height = 6) {
  grDevices::png(filename = file, width = width, height = height,
                 units = "in", res = 600, type = "cairo")
  on.exit(grDevices::dev.off(), add = TRUE)
  plot_fn()
}

# ----------------------------------------------------------------------------
# Core training pipeline for ONE endpoint (binary target). Shared by the
# single-endpoint run (Model Setup tab) and the multi-endpoint batch module,
# so the two modes can never diverge methodologically.
#
# Returns: models (fitted objects), metrics (long train/test frame),
# roc (test ROC objects), log (training log lines), plus the raw-unit training
# predictors and outcome needed downstream by the white-box tree.
# ----------------------------------------------------------------------------
train_pipeline <- function(df, target, preds, balance, train_prop, seed,
                           tunelen, model_labels) {
  set.seed(seed)
  log_msgs <- character(0)

  # --- outcome: positive class = "Yes" (first level) --------------------
  y_all <- factor(as.character(df[[target]]), levels = c("Yes", "No"))
  keep  <- !is.na(y_all)
  df    <- df[keep, , drop = FALSE]
  y_all <- y_all[keep]

  # --- predictor frame with manual 0/1 dummy coding (NAs preserved) -----
  X_all <- as.data.frame(df[, preds, drop = FALSE])
  X_enc <- data.frame(row.names = seq_len(nrow(X_all)))
  for (v in names(X_all)) {
    x <- X_all[[v]]
    if (is.factor(x) || is.character(x)) {
      x <- factor(x)
      for (lv in levels(x)[-1]) {               # first level = reference
        X_enc[[paste0(v, "_", make.names(lv))]] <- as.integer(x == lv)
      }
    } else {
      X_enc[[v]] <- as.numeric(x)
    }
  }

  # --- stratified train/test split --------------------------------------
  idx <- caret::createDataPartition(y_all, p = train_prop, list = FALSE)
  X_train <- X_enc[idx, , drop = FALSE];  X_test <- X_enc[-idx, , drop = FALSE]
  y_train <- y_all[idx];                  y_test  <- y_all[-idx]

  # --- leakage-free K-NN imputation -------------------------------------
  # caret executes `sampling` (SMOTE/ROSE/up/down) inside each resample
  # BEFORE `preProcess`, and themis::smote()/rose() refuse missing values.
  # Imputation therefore precedes train(), but is FIT ON THE TRAINING
  # PARTITION ONLY and merely *applied* to the test set — no leakage.
  # NOTE: caret's knnImpute silently centers+scales the output. Keep a raw
  # copy of the training predictors so the white-box tree can be trained in
  # ORIGINAL clinical units (degrees, years) for interpretability.
  X_train_raw <- X_train
  pp_imp  <- caret::preProcess(X_train, method = "knnImpute")
  X_train <- predict(pp_imp, X_train)
  X_test  <- predict(pp_imp, X_test)

  # --- resampling & balancing -------------------------------------------
  sampling_arg <- switch(balance,
    "none" = NULL, "smote" = "smote", "rose" = "rose",
    "up"   = "up", "down"  = "down")

  ctrl <- caret::trainControl(
    method          = "cv",
    number          = 5,
    classProbs      = TRUE,
    summaryFunction = twoClassSummary,
    sampling        = sampling_arg,
    savePredictions = "final",
    allowParallel   = FALSE
  )
  # fallback control without balancing, used if a model fails under sampling
  ctrl_nosamp <- ctrl
  ctrl_nosamp$sampling <- NULL

  train_one <- function(m, use_ctrl) {
    args <- c(list(x = X_train, y = y_train, method = m,
                   trControl  = use_ctrl,
                   metric     = "ROC",
                   tuneLength = tunelen,
                   preProcess = c("center", "scale")),
              model_extra_args(m))
    do.call(caret::train, args)
  }

  models    <- list()
  # pre-structured empty frame: keeps column names even if every model fails
  metric_df <- data.frame(
    Model = character(), Set = character(), Accuracy = numeric(),
    F1 = numeric(), Sensitivity = numeric(), Specificity = numeric(),
    ROC_AUC = numeric(), stringsAsFactors = FALSE)
  roc_list  <- list()
  test_pred_list <- list()   # held-out predictions per model, for CSV export
  n_models  <- length(model_labels)

  withProgress(message = paste("Training models —", target), value = 0, {
    for (label in model_labels) {
      m  <- MODEL_REGISTRY[[label]]
      incProgress(1 / n_models, detail = label)

      first_err <- NULL
      if (identical(m, "xgbDirect")) {
        # direct xgboost engine (bypasses the broken caret xgbTree wrapper);
        # balancing is already applied inside each fold by the engine itself
        fit <- tryCatch(
          train_xgb_direct(X_train, y_train, balance = balance,
                           tune_len = tunelen, seed = seed),
          error = function(e) { first_err <<- conditionMessage(e); NULL })
      } else {
        fit <- tryCatch(
          train_one(m, ctrl),
          error = function(e) { first_err <<- conditionMessage(e); NULL })

        # automatic retry without balancing when the sampled run failed
        if (is.null(fit) && !is.null(sampling_arg)) {
          fit <- tryCatch(
            train_one(m, ctrl_nosamp),
            error = function(e) NULL)
          if (!is.null(fit)) {
            log_msgs <- c(log_msgs, paste0(
              "[RETRY] ", label, ": failed under ", balance,
              " (", first_err, ") — trained without balancing."))
          }
        }
      }
      if (is.null(fit)) {
        log_msgs <- c(log_msgs, paste0("[ERROR] ", label, ": ", first_err))
        next
      }

      # predictions (caret re-applies the stored preProcess internally)
      # NOTE: plain `<-` is used on purpose. All of these objects live in
      # THIS frame; combining `<<-` with a complex assignment such as
      # `roc_list[[label]] <<- ...` makes R look the name up in the PARENT
      # environments and fails with "object 'roc_list' not found".
      pred_err <- tryCatch({
        pr_tr_c <- predict(fit, X_train)
        pr_tr_p <- predict(fit, X_train, type = "prob")[["Yes"]]
        pr_te_c <- predict(fit, X_test)
        pr_te_p <- predict(fit, X_test,  type = "prob")[["Yes"]]

        metric_df <- rbind(
          metric_df,
          compute_metrics(y_train, pr_tr_c, pr_tr_p, "Train", label),
          compute_metrics(y_test,  pr_te_c, pr_te_p, "Test",  label)
        )
        roc_list[[label]] <- tryCatch(
          pROC::roc(y_test, pr_te_p, quiet = TRUE),
          error = function(e) NULL)
        models[[label]] <- fit
        test_pred_list[[label]] <- list(prob = pr_te_p, class = pr_te_c)
        NULL
      }, error = function(e) conditionMessage(e))
      if (!is.null(pred_err)) {
        log_msgs <- c(log_msgs, paste0("[ERROR] ", label,
                                       " (prediction): ", pred_err))
        next
      }
      log_msgs <- c(log_msgs, paste0("[OK] ", label, " — CV ROC = ",
                                     round(max(fit$results$ROC, na.rm = TRUE), 3)))
    }
  })

  # held-out test-set predictions, one ProbYes/Class pair per model
  test_pred_df <- data.frame(Observed = y_test)
  for (lb in names(test_pred_list)) {
    test_pred_df[[paste0(lb, "_ProbYes")]] <- round(test_pred_list[[lb]]$prob, 3)
    test_pred_df[[paste0(lb, "_Class")]]   <- as.character(test_pred_list[[lb]]$class)
  }

  list(models = models, metrics = metric_df, roc = roc_list, log = log_msgs,
       X_train_raw = X_train_raw, y_train = y_train,
       n_train = nrow(X_train), n_test = nrow(X_test),
       pp_imp = pp_imp,                 # training-fitted imputer (for new cases)
       feature_names = names(X_train),  # training column order (for new cases)
       predictors = preds,              # predictors used AT TRAINING TIME
       test_pred = test_pred_df)
}

# ============================================================================
# 4. UI
# ============================================================================
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Calcaneus Morphometry ML Dashboard",
                  titleWidth = 340),

  dashboardSidebar(
    width = 340,
    sidebarMenu(
      id = "tabs",
      menuItem("Data Overview",      tabName = "data",    icon = icon("table")),
      menuItem("Model Setup",        tabName = "setup",   icon = icon("sliders-h")),
      menuItem("Performance Metrics",tabName = "metrics", icon = icon("chart-line")),
      menuItem("ROC Curves",         tabName = "roc",     icon = icon("chart-area")),
      menuItem("Train vs Test",      tabName = "compare", icon = icon("chart-bar")),
      menuItem("Feature Importance", tabName = "fimp",    icon = icon("sort-amount-down")),
      menuItem("White-Box Tree",     tabName = "tree",    icon = icon("sitemap")),
      menuItem("Correlations",       tabName = "corr",    icon = icon("project-diagram")),
      menuItem("Prediction (New Case)", tabName = "predict", icon = icon("stethoscope")),
      menuItem("Multi-Endpoint Batch", tabName = "batch", icon = icon("layer-group")),
      menuItem("About & Documentation", tabName = "about", icon = icon("book"))
    ),
    hr(),
    fileInput("upload", "Upload raw data (.xlsx / .csv, optional)",
              accept = c(".xlsx", ".xls", ".csv")),
    helpText("If no file is uploaded, the bundled 'calcaneus_clean.csv'",
             "produced by data_prep.R is used.")
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper {background-color: #f7f9fb;}
      .box {border-top: 3px solid #2c7fb8;}
      h3 {color:#22578a;}
    "))),
    tabItems(

      # ---- Tab: Data overview -------------------------------------------
      tabItem(tabName = "data",
        fluidRow(
          box(width = 12, title = "Dataset after standardization",
              status = "primary", solidHeader = TRUE,
              DTOutput("data_table"))
        ),
        fluidRow(
          valueBoxOutput("vb_n",     width = 3),
          valueBoxOutput("vb_miss",  width = 3),
          valueBoxOutput("vb_class", width = 6)
        ),
        fluidRow(
          box(width = 12, title = "Variable summary", status = "info",
              solidHeader = TRUE, verbatimTextOutput("data_summary"))
        )
      ),

      # ---- Tab: Model setup ---------------------------------------------
      tabItem(tabName = "setup",
        fluidRow(
          box(width = 5, title = "Experimental design", status = "primary",
              solidHeader = TRUE,
              selectInput("target", "Target (binary clinical outcome):",
                          choices = NULL),
              checkboxGroupInput("predictors", "Predictor variables:",
                                 choices = NULL),
              hr(),
              radioButtons("balance", "Class-imbalance handling:",
                           choices = c("None"              = "none",
                                       "SMOTE"             = "smote",
                                       "ROSE"              = "rose",
                                       "Up-sampling"       = "up",
                                       "Down-sampling"     = "down"),
                           selected = "smote"),
              sliderInput("train_prop", "Training-set proportion:",
                          min = 0.6, max = 0.9, value = 0.75, step = 0.05),
              numericInput("seed", "Random seed:", value = 2025, min = 1),
              numericInput("tunelen", "caret tuneLength (grid size):",
                           value = 3, min = 1, max = 10),
              actionButton("train_btn", "Train 10 Models (5-Fold CV)",
                           class = "btn-primary btn-lg", icon = icon("play"))
          ),
          box(width = 7, title = "Pipeline specification", status = "info",
              solidHeader = TRUE,
              tags$ul(
                tags$li(strong("Pre-processing (leakage-free):"),
                        " K-NN imputation is fitted on the training partition ",
                        "only and merely applied to the test set; centering ",
                        "and scaling are fitted inside every cross-validation ",
                        "resample."),
                tags$li(strong("Resampling:"), " stratified 5-fold ",
                        "cross-validation (twoClassSummary, class probabilities)."),
                tags$li(strong("Balancing:"), " applied within resamples so ",
                        "synthetic observations never leak into validation folds."),
                tags$li(strong("Algorithms:"), paste(names(MODEL_REGISTRY),
                                                     collapse = ", "))
              ),
              hr(),
              h4("Training log"),
              verbatimTextOutput("train_log")
          )
        )
      ),

      # ---- Tab: Metrics --------------------------------------------------
      tabItem(tabName = "metrics",
        fluidRow(
          box(width = 12, title = "Train / Test performance of all models",
              status = "primary", solidHeader = TRUE,
              downloadButton("dl_metrics", "Download metrics (CSV)",
                             class = "btn-success"),
              downloadButton("dl_test_pred",
                             "Download test-set predictions (CSV)",
                             class = "btn-info"),
              br(), br(),
              DTOutput("metrics_table"))
        )
      ),

      # ---- Tab: ROC ------------------------------------------------------
      tabItem(tabName = "roc",
        fluidRow(
          box(width = 12, title = "ROC curves on the held-out test set",
              status = "primary", solidHeader = TRUE,
              downloadButton("dl_roc", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("roc_plot", height = "560px"))
        )
      ),

      # ---- Tab: Train vs Test -------------------------------------------
      tabItem(tabName = "compare",
        fluidRow(
          box(width = 12, title = "Train vs. test comparison (grouped bars)",
              status = "primary", solidHeader = TRUE,
              selectInput("cmp_metric", "Metric:",
                          choices = c("Accuracy", "F1", "Sensitivity",
                                      "Specificity", "ROC_AUC"),
                          selected = "ROC_AUC", width = "220px"),
              downloadButton("dl_cmp", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("cmp_plot", height = "520px"))
        )
      ),

      # ---- Tab: Feature importance --------------------------------------
      tabItem(tabName = "fimp",
        fluidRow(
          box(width = 12, title = "Variable importance", status = "primary",
              solidHeader = TRUE,
              selectInput("fimp_model", "Model:", choices = names(MODEL_REGISTRY),
                          width = "260px"),
              downloadButton("dl_fimp", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("fimp_plot", height = "480px"))
        )
      ),

      # ---- Tab: White-box tree -------------------------------------------
      tabItem(tabName = "tree",
        fluidRow(
          box(width = 12, title = "White-box decision tree (CART)",
              status = "primary", solidHeader = TRUE,
              helpText("Interpretable surrogate trained with the rpart ",
                       "algorithm on the same training partition."),
              downloadButton("dl_tree", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("tree_plot", height = "560px"))
        )
      ),

      # ---- Tab: Correlations -------------------------------------------------
      tabItem(tabName = "corr",
        fluidRow(
          box(width = 12, title = "Correlations between morphometric variables",
              status = "primary", solidHeader = TRUE,
              helpText("Pairwise correlations among all numeric variables of the ",
                       "loaded dataset, computed on the complete cases."),
              fluidRow(
                column(4, selectInput("corr_method", "Method:",
                         choices = c("Pearson" = "pearson",
                                     "Spearman" = "spearman"))),
                column(4, checkboxInput("corr_show", "Show coefficients", TRUE)),
                column(4, radioButtons("corr_sig", "Display:",
                         choices = c("All pairs" = "all",
                                     "Only significant (p<0.05)" = "sig"),
                         inline = TRUE))
              ),
              downloadButton("dl_corr",     "Download (600 DPI PNG)",
                             class = "btn-success"),
              downloadButton("dl_corr_csv", "Download table (CSV)",
                             class = "btn-info"),
              plotOutput("corr_plot", height = "560px"))
        )
      ),

      # ---- Tab: Prediction (new case) --------------------------------------
      tabItem(tabName = "predict",
        fluidRow(
          box(width = 5, title = "New case inputs", status = "primary",
              solidHeader = TRUE,
              helpText("Enter the clinical measurements of a new case.",
                       "Empty numeric fields are K-NN imputed with the",
                       "training-fitted imputer, exactly like the test set.",
                       "Train the models on the 'Model Setup' tab first."),
              helpText(tags$em("Note: the selected binary target is excluded",
                               "from the inputs and becomes the outcome to",
                               "predict.")),
              uiOutput("newcase_inputs"),
              hr(),
              selectInput("pred_model", "Model for the headline prediction:",
                          choices = NULL, width = "280px"),
              actionButton("predict_btn", "Predict",
                           class = "btn-primary btn-lg",
                           icon = icon("stethoscope"))
          ),
          box(width = 7, title = "Prediction result", status = "info",
              solidHeader = TRUE,
              uiOutput("pred_result"),
              uiOutput("pred_errors_ui"),
              helpText("Research use only - not a clinical diagnosis."),
              hr(),
              downloadButton("dl_pred", "Download predictions (CSV)",
                             class = "btn-success"),
              br(), br(),
              DTOutput("pred_all_models"))
        )
      ),

      # ---- Tab: Multi-endpoint batch -------------------------------------
      # Runs the SAME pipeline (split, imputation, balancing, 5-fold CV)
      # across ALL binary endpoints and combines the results into single
      # composite figures, so one figure covers every endpoint — solving the
      # journal figure-count limit.
      tabItem(tabName = "batch",
        fluidRow(
          box(width = 4, title = "Batch design", status = "primary",
              solidHeader = TRUE,
              checkboxGroupInput("batch_targets",
                                 "Endpoints (binary targets):",
                                 choices = NULL),
              checkboxGroupInput("batch_models", "Models to compare:",
                                 choices = names(MODEL_REGISTRY),
                                 selected = c("Logistic Regression",
                                              "Random Forest",
                                              "XGBoost",
                                              "SVM (RBF)")),
              helpText("Uses the train/test proportion, balancing method,",
                       "seed, tuneLength and predictor selection from the",
                       "'Model Setup' tab. The endpoint under evaluation is",
                       "automatically excluded from its own predictor set."),
              actionButton("batch_btn", "Run Batch Across Endpoints",
                           class = "btn-primary btn-lg",
                           icon = icon("layer-group"))
          ),
          box(width = 8, title = "Consolidated metrics (endpoints &times; models)",
              status = "info", solidHeader = TRUE,
              downloadButton("dl_batch_metrics",
                             "Download consolidated metrics (CSV)",
                             class = "btn-success"),
              br(), br(),
              DTOutput("batch_metrics_table"))
        ),
        fluidRow(
          box(width = 7, status = "primary", solidHeader = TRUE,
              title = "Composite ROC panel &mdash; one figure for ALL endpoints",
              downloadButton("dl_batch_roc", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("batch_roc_plot", height = "640px")),
          box(width = 5, status = "primary", solidHeader = TRUE,
              title = "Test ROC-AUC heatmap &mdash; endpoints &times; models",
              downloadButton("dl_batch_heat", "Download (600 DPI PNG)",
                             class = "btn-success"),
              plotOutput("batch_heat_plot", height = "520px"))
        )
      ),

      # ---- Tab: About ----------------------------------------------------
      tabItem(tabName = "about",
        fluidRow(
          box(width = 12, title = "About & Documentation", status = "primary",
              solidHeader = TRUE,
              uiOutput("about_ui"))
        )
      )
    )
  )
)

# ============================================================================
# 5. SERVER
# ============================================================================
server <- function(input, output, session) {

  # ---- Reactive data source ----------------------------------------------
  raw_data <- reactive({
    if (!is.null(input$upload)) {
      ext <- tolower(tools::file_ext(input$upload$name))
      df <- switch(ext,
        "xlsx" = readxl::read_excel(input$upload$datapath),
        "xls"  = readxl::read_excel(input$upload$datapath),
        "csv"  = readr::read_csv(input$upload$datapath, show_col_types = FALSE),
        stop("Unsupported file type."))
      clean_dataset(as.data.frame(df))
    } else if (file.exists("calcaneus_clean.csv")) {
      clean_dataset(readr::read_csv("calcaneus_clean.csv", show_col_types = FALSE))
    } else if (file.exists("HAM_VERI.xlsx")) {
      clean_dataset(as.data.frame(readxl::read_excel("HAM_VERI.xlsx")))
    } else {
      validate(need(FALSE, "No dataset found. Upload a file or place ",
                    "'calcaneus_clean.csv' next to app.R."))
    }
  })

  binary_cols <- reactive({
    df <- raw_data()
    names(df)[vapply(df, function(x) is.factor(x) && nlevels(x) == 2,
                     logical(1))]
  })

  # Populate the target choices ONCE, when the data source changes.
  # Using observeEvent(raw_data(), ...) is essential: a plain observe() would
  # re-fire whenever input$target changes (it is referenced in the predictor
  # list below) and would keep resetting the user's selection back to "hd_r".
  observeEvent(raw_data(), {
    bc <- binary_cols()
    df <- raw_data()
    # Set target (default: hd_r if present)
    updateSelectInput(session, "target", choices = bc,
                      selected = if ("hd_r" %in% bc) "hd_r" else bc[1])
    # Initialize predictors with sensible defaults, excluding the target
    target_now <- isolate(input$target)
    if (is.null(target_now) || !target_now %in% bc) target_now <- bc[1]
    preds <- setdiff(names(df), target_now)
    default_preds <- intersect(c("sex","age","ga_r","ga_l","ba_r","ba_l"),
                               preds)
    updateCheckboxGroupInput(session, "predictors", choices = preds,
                             selected = default_preds)
    # batch module: offer every binary endpoint, all selected by default
    updateCheckboxGroupInput(session, "batch_targets", choices = bc,
                             selected = bc)
  }, once = TRUE)

  # When the TARGET changes, refresh the predictor list (exclude the target)
  # while preserving any valid predictor selection the user already made.
  observeEvent(input$target, {
    req(input$target)
    df    <- raw_data()
    preds <- setdiff(names(df), input$target)
    current <- intersect(input$predictors, preds)
    default_preds <- intersect(c("sex","age","ga_r","ga_l","ba_r","ba_l"),
                               preds)
    updateCheckboxGroupInput(session, "predictors", choices = preds,
                             selected = if (length(current) > 0) current
                                        else default_preds)
  }, ignoreInit = TRUE)

  # ---- Data overview ------------------------------------------------------
  output$data_table <- renderDT({
    datatable(raw_data(), options = list(pageLength = 12, scrollX = TRUE),
              rownames = FALSE, class = "stripe hover")
  })

  output$vb_n <- renderValueBox({
    valueBox(nrow(raw_data()), "Observations", icon = icon("users"),
             color = "blue")
  })
  output$vb_miss <- renderValueBox({
    valueBox(sum(is.na(raw_data())), "Missing cells (imputed, train-fitted)",
             icon = icon("puzzle-piece"), color = "yellow")
  })
  output$vb_class <- renderValueBox({
    req(input$target)
    tb <- table(raw_data()[[input$target]], useNA = "no")
    valueBox(paste(names(tb), as.integer(tb), sep = ": ", collapse = "  |  "),
             paste("Class balance of target:", input$target),
             icon = icon("balance-scale"), color = "purple")
  })

  output$data_summary <- renderPrint({ summary(raw_data()) })

  # ---- Training ------------------------------------------------------------
  results <- eventReactive(input$train_btn, {

    df <- raw_data()
    req(input$target, input$predictors)
    validate(need(length(input$predictors) >= 1,
                  "Select at least one predictor."))

    res <- train_pipeline(df, input$target, input$predictors, input$balance,
                          input$train_prop, input$seed, input$tunelen,
                          names(MODEL_REGISTRY))

    metric_df <- res$metrics
    log_msgs  <- res$log


    # ---- white-box surrogate tree (CART) -----------------------------------
    # The tree is a VISUALIZATION aid, not a scoring model, so two different
    # choices are made here versus the main pipeline:
    #   (1) It is trained in ORIGINAL clinical units: caret's knnImpute
    #       silently centers+scales its output, which would produce
    #       uninterpretable split thresholds (e.g. "ba_l < -0.46"). For the
    #       tree we therefore median-impute the RAW training predictors, so
    #       splits read e.g. "ba_l < 27" (degrees) — citable in a manuscript.
    #   (2) It is trained on the class-BALANCED data so the minority class is
    #       visible to the algorithm, then aggressively pruned (1-SE rule,
    #       hard cap of 10 leaves) so the figure stays readable in print.
    tree_fit <- tryCatch({
      X_tree <- res$X_train_raw
      for (v in names(X_tree)) {
        med <- stats::median(X_tree[[v]], na.rm = TRUE)
        X_tree[[v]][is.na(X_tree[[v]])] <- med
      }
      # SMOTE/ROSE interpolate 0/1 dummy predictors into fractional values
      # (e.g. sex_Male = 0.0032), producing uninterpretable split thresholds.
      # Interpolated dummy rows are also scientifically dubious. Instead of
      # post-hoc rounding, balance with CLASS-WEIGHTED RANDOM OVER-SAMPLING:
      # real rows only, drawn with replacement, so binaries stay strictly 0/1
      # and numeric predictors keep their original values.
      ytb <- table(res$y_train)
      if (length(ytb) == 2 && min(ytb) >= 1 && !identical(as.integer(ytb[1]),
                                                          as.integer(ytb[2]))) {
        minc <- names(ytb)[which.min(ytb)]
        target_n <- max(ytb)
        need <- target_n - min(ytb)
        min_idx <- which(res$y_train == minc)
        dup_idx <- sample(min_idx, need, replace = TRUE)
        X_tree  <- rbind(X_tree, X_tree[dup_idx, , drop = FALSE])
        y_tree  <- c(res$y_train, res$y_train[dup_idx])
      } else {
        y_tree <- res$y_train
      }
      tree_df  <- data.frame(.outcome = y_tree, X_tree,
                             check.names = FALSE)
      fit <- rpart::rpart(
        .outcome ~ ., data = tree_df, method = "class",
        parms   = list(loss = matrix(c(0, 3, 1, 0), nrow = 2, byrow = TRUE)),
        control = rpart.control(cp = 0.001, minbucket = 15, maxdepth = 6,
                                xval = 10, maxsurrogate = 0, usesurrogate = 0))
      # 1-SE rule: smallest tree whose CV error is within 1 SE of the minimum
      cp_tab <- fit$cptable
      if (!is.null(cp_tab) && nrow(cp_tab) > 1) {
        i_min <- which.min(cp_tab[, "xerror"])
        thr   <- cp_tab[i_min, "xerror"] + cp_tab[i_min, "xstd"]
        ok    <- which(cp_tab[, "xerror"] <= thr)
        fit   <- rpart::prune(fit, cp = cp_tab[ok[1], "CP"])
        # hard cap: at most 10 terminal leaves, for figure readability
        cps <- sort(unique(fit$cptable[, "CP"]))
        ci  <- 1L
        while (sum(fit$frame$var == "<leaf>") > 10 && ci <= length(cps)) {
          fit <- rpart::prune(fit, cp = cps[ci]); ci <- ci + 1L
        }
      }
      fit
    }, error = function(e) {
      log_msgs <<- c(log_msgs, paste0("[TREE ERROR] ", conditionMessage(e)))
      NULL
    })

    # ---- overfitting analysis ------------------------------------------------
    # Compute Train-Test ROC-AUC gap for each model and flag severity.
    # This is CRITICAL for publication: reviewers expect explicit quantification
    # of memorization vs generalization, not just final test metrics.
    metric_df <- metric_df %>%
      dplyr::mutate(Train_Test_Gap = NA_real_, Overfit_Flag = NA_character_)

    for (lbl in unique(metric_df$Model)) {
      tr <- metric_df[metric_df$Model == lbl & metric_df$Set == "Train", ]
      te <- metric_df[metric_df$Model == lbl & metric_df$Set == "Test", ]
      if (nrow(tr) > 0 && nrow(te) > 0) {
        gap <- round(tr$ROC_AUC - te$ROC_AUC, 3)
        flag <- ifelse(gap > 0.25, "HIGH",
                ifelse(gap > 0.15, "MODERATE",
                ifelse(gap > 0.10, "MILD", "OK")))
        metric_df$Train_Test_Gap[metric_df$Model == lbl] <- gap
        metric_df$Overfit_Flag[metric_df$Model == lbl] <- flag
      }
    }

    # Reorder columns for publication table
    metric_df <- metric_df %>%
      dplyr::select(Model, Set, Accuracy, F1, Sensitivity, Specificity,
                    ROC_AUC, Train_Test_Gap, Overfit_Flag)

    # Summary for training log (transparent reporting)
    overfit_summary <- metric_df %>%
      dplyr::filter(!is.na(Train_Test_Gap)) %>%
      dplyr::distinct(Model, Train_Test_Gap, Overfit_Flag) %>%
      dplyr::arrange(dplyr::desc(Train_Test_Gap))

    overfit_msg <- paste0(
      "\n", strrep("=", 70), "\n",
      "OVERFITTING ANALYSIS (Train - Test ROC-AUC gap)\n",
      strrep("-", 70), "\n",
      paste0(sprintf("%-25s gap = %5.3f  [%s]",
                     overfit_summary$Model,
                     overfit_summary$Train_Test_Gap,
                     overfit_summary$Overfit_Flag),
             collapse = "\n"),
      "\n", strrep("-", 70), "\n",
      "Interpretation: HIGH (>0.25) = severe memorization;\n",
      "  MODERATE (0.15-0.25) = some overfitting;\n",
      "  MILD (0.10-0.15) = acceptable; OK (<=0.10) = good generalization.\n",
      "For publication, report BOTH train and test metrics and discuss gaps.\n",
      strrep("=", 70)
    )

    list(models = res$models, metrics = metric_df, roc = res$roc,
         tree = tree_fit, log = c(log_msgs, overfit_msg),
         n_train = res$n_train, n_test = res$n_test,
         pp_imp = res$pp_imp, feature_names = res$feature_names,
         predictors = res$predictors, X_train_raw = res$X_train_raw,
         test_pred = res$test_pred)
  })

  # Reproducibility tag embedded in EVERY download filename: train/test
  # proportion, caret tuneLength and random seed, so any figure/table can be
  # traced back to the exact experimental settings that produced it.
  file_tag <- reactive({
    sprintf("train%g_tune%s_seed%s",
            round(input$train_prop * 100), input$tunelen, input$seed)
  })

  # ---- Training log --------------------------------------------------------
  output$train_log <- renderPrint({
    r <- results()
    cat("Training set:", r$n_train, " | Test set:", r$n_test, "\n")
    cat("Balancing   :", input$balance, "| CV: 5-fold | tuneLength:",
        input$tunelen, "\n")
    cat(strrep("-", 60), "\n")
    cat(paste(r$log, collapse = "\n"))
  })

  metrics_rounded <- reactive({
    r <- results()
    m <- r$metrics
    validate(need(nrow(m) > 0,
                  "No model completed successfully. Check the training log ",
                  "on the 'Model Setup' tab."))
    num_cols <- intersect(c("Accuracy","F1","Sensitivity","Specificity",
                            "ROC_AUC"), names(m))
    m[num_cols] <- lapply(m[num_cols], function(z) round(z, 3))
    m
  })

  # ---- Metrics table + CSV -------------------------------------------------
  output$metrics_table <- renderDT({
    datatable(metrics_rounded(), rownames = FALSE, class = "stripe hover",
              options = list(pageLength = 20, dom = "tip")) %>%
      formatStyle("Set", target = "row",
                  backgroundColor = styleEqual("Test", "#eef6fc"))
  })

  output$dl_metrics <- downloadHandler(
    filename = function() paste0("model_metrics_", input$target, "_",
                                 file_tag(), "_", Sys.Date(), ".csv"),
    content  = function(file) write.csv(metrics_rounded(), file,
                                        row.names = FALSE)
  )

  # held-out test-set predictions of every model (Observed + ProbYes + Class)
  output$dl_test_pred <- downloadHandler(
    filename = function() paste0("test_predictions_", input$target, "_",
                                 file_tag(), "_", Sys.Date(), ".csv"),
    content  = function(file) write.csv(results()$test_pred, file,
                                        row.names = FALSE)
  )

  # ---- Prediction (new case) -----------------------------------------------
  # dynamic input widgets, one per predictor (type-aware). After a training
  # run the widgets follow the TRAINING-TIME predictor set stored in
  # results(), so the form always matches what the fitted models expect.
  output$newcase_inputs <- renderUI({
    df <- raw_data()
    # before the first training run, results() raises Shiny's silent
    # "not ready yet" error — catch it so the form still renders
    r <- tryCatch(results(), error = function(e) NULL)
    preds <- if (!is.null(r) && !is.null(r$predictors)) r$predictors
             else input$predictors
    req(preds)
    lapply(preds, function(v) {
      x <- df[[v]]
      if (is.factor(x)) {
        selectInput(paste0("newcase_", v), label = v,
                    choices = levels(x), width = "220px")
      } else {
        med <- suppressWarnings(stats::median(as.numeric(x), na.rm = TRUE))
        if (!is.finite(med)) med <- NA_real_
        numericInput(paste0("newcase_", v), label = v, value = med,
                     width = "220px")
      }
    })
  })

  # refresh the model selector after every training run; default = best
  # test ROC-AUC
  observeEvent(results(), {
    r <- results()
    ms <- names(r$models)
    if (length(ms) == 0) return()
    te <- r$metrics[r$metrics$Set == "Test", , drop = FALSE]
    te <- te[!is.na(te$ROC_AUC), , drop = FALSE]
    best <- if (nrow(te) > 0) te$Model[which.max(te$ROC_AUC)] else ms[1]
    updateSelectInput(session, "pred_model", choices = ms,
                      selected = if (best %in% ms) best else ms[1])
  }, ignoreInit = TRUE)

  prediction <- eventReactive(input$predict_btn, {
    r <- results()
    validate(need(length(r$models) > 0,
                  "Train the models first on the 'Model Setup' tab."),
             need(!is.null(r$feature_names) && length(r$feature_names) > 0,
                  "Training feature set is missing; re-train the models."))
    df <- raw_data()

    # Build the new case from the predictors used AT TRAINING TIME (stored in
    # the results object), not from the live checkbox selection — otherwise a
    # selection change after training silently breaks the feature mapping.
    train_preds <- r$predictors
    pred_errors <- character(0)

    # one-row raw predictor frame, encoded exactly like the training pipeline
    new_row <- data.frame(row.names = 1L)
    for (v in train_preds) {
      x   <- df[[v]]
      val <- input[[paste0("newcase_", v)]]
      if (is.null(val)) val <- NA          # input not rendered -> impute later
      if (is.factor(x)) {
        x <- factor(x)
        for (lv in levels(x)[-1])          # first level = reference
          new_row[[paste0(v, "_", make.names(lv))]] <-
            if (is.na(val)) NA_integer_ else as.integer(val == lv)
      } else {
        new_row[[v]] <- suppressWarnings(as.numeric(val))   # NA -> imputed
      }
    }

    # leakage-free imputation with the imputer FITTED ON THE TRAINING SET.
    # The imputed frame is EXACTLY what the models saw during training and
    # testing (caret stores its center/scale preProcess inside each fit and
    # re-applies it internally), so it is fed to predict() unchanged.
    new_imp <- tryCatch(predict(r$pp_imp, new_row),
                        error = function(e) {
                          pred_errors <<- c(pred_errors,
                            paste0("[impute] ", conditionMessage(e)))
                          new_row
                        })
    # if the imputer failed (e.g. an all-NA column was dropped during
    # training), fall back to median imputation so no NA ever reaches a model
    for (v in names(new_imp)) {
      if (is.na(new_imp[[v]][1])) {
        med <- stats::median(r$X_train_raw[[v]], na.rm = TRUE)
        if (!is.finite(med)) med <- 0
        new_imp[[v]][1] <- med
      }
    }
    # align columns to the training feature set; any feature missing from the
    # new case is filled with its training median (never left as NA)
    for (mc in setdiff(r$feature_names, names(new_imp))) {
      med <- stats::median(r$X_train_raw[[mc]], na.rm = TRUE)
      if (!is.finite(med)) med <- 0
      new_imp[[mc]] <- med
    }
    new_imp <- new_imp[, r$feature_names, drop = FALSE]

    rows <- list()
    for (lb in names(r$models)) {
      fit <- r$models[[lb]]
      pr <- tryCatch(
        suppressWarnings(predict(fit, new_imp, type = "prob")[["Yes"]]),
        error = function(e) {
          pred_errors <<- c(pred_errors, paste0("[", lb, "] ", conditionMessage(e)))
          NA_real_
        })
      cl <- tryCatch(
        suppressWarnings(as.character(predict(fit, new_imp))),
        error = function(e) {
          pred_errors <<- c(pred_errors, paste0("[", lb, " class] ", conditionMessage(e)))
          NA_character_
        })
      rows[[lb]] <- data.frame(Model = lb,
                               Prob_Yes = if (length(pr)) round(pr, 3) else NA_real_,
                               Prediction = if (length(cl)) cl else NA_character_,
                               stringsAsFactors = FALSE)
    }
    out <- dplyr::bind_rows(rows)
    attr(out, "errors") <- pred_errors
    out
  })

  # surface silent per-model prediction failures in the UI
  output$pred_errors_ui <- renderUI({
    pm <- prediction()
    errs <- attr(pm, "errors")
    if (is.null(errs) || length(errs) == 0) return(NULL)
    helpText(tags$span(style = "color:#a94442;",
                       paste("Model prediction warnings:",
                             paste(errs, collapse = " | "))))
  })

  output$pred_result <- renderUI({
    pm <- prediction()
    req(input$pred_model)
    row <- pm[pm$Model == input$pred_model, , drop = FALSE]
    validate(need(nrow(row) > 0 && !is.na(row$Prob_Yes[1]),
                  "Prediction unavailable for this model."))
    cls <- row$Prediction[1]
    pr  <- row$Prob_Yes[1]
    col <- if (identical(cls, "Yes")) "#c0392b" else "#1e8449"
    tagList(
      h3(style = paste0("color:", col, ";"),
         paste0(input$target, ": ", cls),
         tags$small(style = "color:#555;",
                    paste0("  (P(Yes) = ", sprintf("%.3f", pr),
                           ", model: ", input$pred_model, ")")))
    )
  })

  output$pred_all_models <- renderDT({
    datatable(prediction(), rownames = FALSE, class = "stripe hover",
              options = list(pageLength = 12, dom = "tip"))
  })

  output$dl_pred <- downloadHandler(
    filename = function() paste0("new_case_prediction_",
                                 MODEL_SHORT[input$pred_model], "_",
                                 input$target, "_", file_tag(), "_",
                                 Sys.Date(), ".csv"),
    content  = function(file) {
      out <- prediction()
      out$Target     <- input$target
      out$Train_Prop <- input$train_prop
      out$TuneLength <- input$tunelen
      out$Seed       <- input$seed
      write.csv(out, file, row.names = FALSE)
    }
  )

  # ---- ROC plot -------------------------------------------------------------
  # Journal-ready: NO title (captions live in the manuscript), and the curves
  # fill the canvas edge-to-edge (square plot region, "internal" axis style,
  # minimal margins) so there is no empty space on the left/right.
  draw_roc <- function() {
    r <- results()
    validate(need(length(r$roc) > 0, "No ROC objects available."))
    cols <- grDevices::hcl.colors(length(r$roc), palette = "Dark 3")
    op <- par(no.readonly = TRUE); on.exit(par(op))
    par(mar = c(4.0, 4.0, 0.4, 0.4), mgp = c(2.4, 0.7, 0), pty = "s")
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i",
         xlab = "1 - Specificity", ylab = "Sensitivity")
    abline(a = 0, b = 1, lty = 3, col = "grey60")
    lg <- character(0)
    for (i in seq_along(r$roc)) {
      roc_i <- r$roc[[i]]
      # manual coordinates: full control over axes and margins
      lines(1 - roc_i$specificities, roc_i$sensitivities,
            col = cols[i], lwd = 2)
      lg <- c(lg, sprintf("%s (AUC = %.3f)", names(r$roc)[i],
                          as.numeric(pROC::auc(roc_i))))
    }
    legend("bottomright", legend = lg, col = cols, lwd = 2,
           cex = 0.7, bty = "n", inset = 0.01)
  }

  output$roc_plot <- renderPlot({ draw_roc() })
  output$dl_roc <- downloadHandler(
    filename = function() paste0("roc_curves_600dpi_", input$target, "_",
                                 file_tag(), "_", Sys.Date(), ".png"),
    content  = function(file) save_base_600dpi(file, draw_roc, 7, 7)
  )

  # ---- Train vs Test comparison --------------------------------------------
  cmp_ggplot <- function() {
    # NOTE: dplyr:: is mandatory here — MASS is loaded after dplyr and its
    # MASS::select() masks dplyr::select(), which would raise
    # "unused arguments (Model, Set, value = ...)".
    m <- metrics_rounded() %>%
      dplyr::select(Model, Set, value = dplyr::all_of(input$cmp_metric)) %>%
      dplyr::mutate(Model = factor(Model, levels = names(MODEL_REGISTRY)))
    m$Label <- MODEL_SHORT[as.character(m$Model)]
    m$Label <- factor(m$Label, levels = MODEL_SHORT[levels(m$Model)])
    ggplot(m, aes(x = Label, y = value, fill = Set)) +
      geom_col(position = position_dodge(width = 0.75), width = 0.65) +
      geom_text(aes(label = sprintf("%.3f", value)),
                position = position_dodge(width = 0.75),
                vjust = -0.4, size = 2.6) +
      scale_fill_manual(values = c(Train = "#9ecae1", Test = "#08519c")) +
      coord_cartesian(ylim = c(0, 1.08)) +
      labs(x = NULL, y = input$cmp_metric) +   # no title: journals want title-free figures
      theme_minimal(base_size = 13) +
      theme(axis.text.x = element_text(angle = 35, hjust = 1),
            legend.position = "top")
  }

  output$cmp_plot <- renderPlot({ print(cmp_ggplot()) })
  output$dl_cmp <- downloadHandler(
    filename = function() paste0("train_vs_test_", input$cmp_metric, "_",
                                 input$target, "_", file_tag(), "_",
                                 Sys.Date(), ".png"),
    content = function(file)
      ggplot2::ggsave(file, cmp_ggplot(), width = 10, height = 6,
                      dpi = 600, device = "png")
  )

  # ---- Feature importance ----------------------------------------------------
  fimp_ggplot <- function() {
    r <- results()
    req(input$fimp_model)
    fit <- r$models[[input$fimp_model]]
    validate(need(!is.null(fit), "This model failed to train; pick another."))
    if (inherits(fit, "xgb_direct")) {
      # gain-based importance from the direct xgboost engine, scaled 0-100
      xi <- xgboost::xgb.importance(feature_names = fit$feature_names,
                                    model = fit$booster)
      validate(need(nrow(xi) > 0,
                    "Variable importance is not available for this model."))
      imp <- data.frame(Variable   = xi$Feature,
                        Importance = xi$Gain / max(xi$Gain) * 100)
    } else {
      vi <- tryCatch(caret::varImp(fit, scale = TRUE)$importance,
                     error = function(e) NULL)
      validate(need(!is.null(vi) && nrow(vi) > 0,
                    "Variable importance is not available for this algorithm."))
      imp <- data.frame(Variable = rownames(vi), Importance = vi[[1]])
    }
    imp <- imp[order(imp$Importance), , drop = FALSE]
    imp$Variable <- factor(imp$Variable, levels = imp$Variable)
    ggplot(imp, aes(x = Variable, y = Importance)) +
      geom_col(fill = "#2c7fb8") +
      coord_flip() +
      labs(x = NULL, y = "Importance (scaled 0–100)") +   # no title: journals want title-free figures
      theme_minimal(base_size = 13)
  }

  output$fimp_plot <- renderPlot({ print(fimp_ggplot()) })
  output$dl_fimp <- downloadHandler(
    filename = function() paste0("feature_importance_",
                                 MODEL_SHORT[input$fimp_model], "_",
                                 input$target, "_", file_tag(), "_",
                                 Sys.Date(), ".png"),
    content = function(file)
      ggplot2::ggsave(file, fimp_ggplot(), width = 8, height = 6,
                      dpi = 600, device = "png")
  )

  # ---- Correlations ---------------------------------------------------------
  # Pairwise Pearson/Spearman correlations with p-values among all numeric
  # variables of the dataset; used for the morphometric correlation analyses
  # reported in the article (e.g. CCA vs. MINH/MAXH/MAXW/BA, PFA vs. CCA/BA).
  corr_results <- reactive({
    req(input$corr_method)
    nums <- raw_data()[, vapply(raw_data(), is.numeric, logical(1)), drop = FALSE]
    validate(need(ncol(nums) >= 2, "Need at least two numeric variables."))
    d <- nums[stats::complete.cases(nums), , drop = FALSE]
    validate(need(nrow(d) >= 3, "Not enough complete observations."))
    cn  <- colnames(d)
    res <- expand.grid(Var1 = cn, Var2 = cn, stringsAsFactors = FALSE)
    res[c("r", "p")] <- t(mapply(function(a, b)
      unlist(stats::cor.test(d[[a]], d[[b]], method = input$corr_method,
                             exact = FALSE)[c("estimate", "p.value")]),
      res$Var1, res$Var2))
    res$Variable1 <- factor(res$Var1, levels = cn)
    res$Variable2 <- factor(res$Var2, levels = cn)
    res
  })

  corr_ggplot <- function() {
    cr <- corr_results()
    show_sig_only <- isTRUE(input$corr_sig == "sig")
    star <- ifelse(cr$p < 0.001, "***",
                   ifelse(cr$p < 0.01, "**",
                          ifelse(cr$p < 0.05, "*", "")))
    if (show_sig_only) {
      cr$rdisp <- ifelse(star == "", NA_real_, cr$r)   # blank non-significant tiles
    } else {
      cr$rdisp <- cr$r
    }
    p <- ggplot(cr, aes(x = Variable2, y = Variable1, fill = rdisp)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                           limits = c(-1, 1), na.value = "grey93",
                           name = "r") +
      labs(x = NULL, y = NULL) +   # no title: journals want title-free figures
      theme_minimal(base_size = 13) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid = element_blank())
    if (isTRUE(input$corr_show)) {
      p <- p + geom_text(aes(label = ifelse(is.na(rdisp), "",
                                            sprintf("%.2f%s", r, star))),
                         size = 3.2)
    }
    p
  }

  output$corr_plot <- renderPlot({ print(corr_ggplot()) })
  output$dl_corr <- downloadHandler(
    filename = function() paste0("correlation_heatmap_", input$corr_method,
                                 "_", file_tag(), "_", Sys.Date(), ".png"),
    content = function(file)
      ggplot2::ggsave(file, corr_ggplot(), width = 9, height = 7,
                      dpi = 600, device = "png")
  )
  output$dl_corr_csv <- downloadHandler(
    filename = function() paste0("correlations_", input$corr_method,
                                 "_", file_tag(), "_", Sys.Date(), ".csv"),
    content = function(file)
      readr::write_csv(corr_results()[, c("Var1", "Var2", "r", "p")], file)
  )

  # ---- White-box tree ----------------------------------------------------------
  draw_tree <- function() {
    r <- results()
    if (is.null(r$tree)) {
      plot.new()
      text(0.5, 0.5, "Decision tree could not be trained.\nCheck the training log on the 'Model Setup' tab.",
           cex = 1.2, col = "darkred", font = 2)
      return()
    }
    # If the tree collapsed to a single node (no splits), show a message
    if (nrow(r$tree$frame) == 1) {
      plot.new()
      text(0.5, 0.5, "Tree collapsed to root node (no splits).\nTry a different balancing method or check class imbalance.",
           cex = 1.2, col = "darkred", font = 2)
      return()
    }
    # no main title: journals want title-free figures.
    # type = 2 prints the split rule INSIDE each interior node instead of on
    # the branches, so split labels can no longer collide with child nodes.
    # ycompress = FALSE keeps every level on its own evenly spaced row (no
    # vertical squeezing); node-number boxes are omitted to declutter.
    rpart.plot::rpart.plot(
      r$tree, type = 2, extra = 104, under = TRUE, faclen = 0,
      cex = 0.9, box.palette = "BuGn", shadow.col = "gray",
      fallen.leaves = TRUE, branch = 0.5,
      tweak = 1.2, ycompress = FALSE,
      roundint = FALSE, digits = 2)
  }

  output$tree_plot <- renderPlot({ draw_tree() })
  output$dl_tree <- downloadHandler(
    filename = function() paste0("decision_tree_600dpi_", input$target, "_",
                                 file_tag(), "_", Sys.Date(), ".png"),
    content  = function(file) save_base_600dpi(file, draw_tree, 10, 7)
  )

  # ---- Multi-endpoint batch -------------------------------------------------
  # Trains the selected models for EVERY selected endpoint with the identical
  # pipeline (train_pipeline is shared with the single-endpoint run), then
  # combines all endpoints into composite figures: one multi-panel ROC figure
  # and one AUC heatmap cover all 7 endpoints within journal figure limits.
  batch_results <- eventReactive(input$batch_btn, {
    df <- raw_data()
    eps <- input$batch_targets
    validate(need(length(eps) >= 1, "Select at least one endpoint."),
             need(length(input$batch_models) >= 1,
                  "Select at least one model."),
             need(length(input$predictors) >= 1,
                  "Select predictors on the 'Model Setup' tab first."))
    out <- list()
    withProgress(message = "Batch training across endpoints", value = 0, {
      for (k in seq_along(eps)) {
        ep <- eps[k]
        incProgress((k - 1) / length(eps), detail = ep)
        # the endpoint itself is never allowed into its own predictor set
        preds <- setdiff(input$predictors, ep)
        out[[ep]] <- tryCatch(
          train_pipeline(df, ep, preds, input$balance, input$train_prop,
                         input$seed, input$tunelen, input$batch_models),
          error = function(e)
            structure(list(error = conditionMessage(e)), class = "batch_error"))
      }
      incProgress(1, detail = "done")
    })
    out
  })

  # consolidated long-format metrics with per-endpoint overfitting gaps
  batch_metrics_long <- reactive({
    br <- batch_results()
    validate(need(length(br) > 0, "Run the batch first."))
    parts <- list()
    for (ep in names(br)) {
      if (inherits(br[[ep]], "batch_error")) next
      m <- br[[ep]]$metrics
      if (nrow(m) == 0) next
      parts[[ep]] <- cbind(Endpoint = ep, m, stringsAsFactors = FALSE)
    }
    validate(need(length(parts) > 0,
                  "Every endpoint failed — see the training log."))
    bm <- do.call(rbind, parts)
    bm$Train_Test_Gap <- NA_real_
    bm$Overfit_Flag   <- NA_character_
    for (ep in unique(bm$Endpoint)) {
      for (lbl in unique(bm$Model)) {
        tr <- bm$Endpoint == ep & bm$Model == lbl & bm$Set == "Train"
        te <- bm$Endpoint == ep & bm$Model == lbl & bm$Set == "Test"
        if (any(tr) && any(te)) {
          gap <- round(bm$ROC_AUC[tr] - bm$ROC_AUC[te], 3)
          flag <- ifelse(gap > 0.25, "HIGH",
                  ifelse(gap > 0.15, "MODERATE",
                  ifelse(gap > 0.10, "MILD", "OK")))
          bm$Train_Test_Gap[tr | te] <- gap
          bm$Overfit_Flag[tr | te]   <- flag
        }
      }
    }
    num_cols <- c("Accuracy","F1","Sensitivity","Specificity","ROC_AUC")
    bm[num_cols] <- lapply(bm[num_cols], function(z) round(z, 3))
    bm
  })

  output$batch_metrics_table <- renderDT({
    datatable(batch_metrics_long(), rownames = FALSE,
              class = "stripe hover",
              options = list(pageLength = 20, scrollX = TRUE)) %>%
      formatStyle("Set", target = "row",
                  backgroundColor = styleEqual("Test", "#eef6fc"))
  })

  output$dl_batch_metrics <- downloadHandler(
    filename = function() paste0("all_endpoints_metrics_", file_tag(), "_",
                                 Sys.Date(), ".csv"),
    content  = function(file) write.csv(batch_metrics_long(), file,
                                        row.names = FALSE)
  )

  # Composite ROC figure: one panel per endpoint, shared model legend.
  # Title-free; panel tags "(a) hd_r" are standard multi-panel labels, not
  # figure titles. Tight margins, curves fill each panel edge-to-edge.
  draw_batch_roc <- function() {
    br <- batch_results()
    validate(need(length(br) > 0, "Run the batch first."))
    eps <- names(br)[!vapply(br, inherits, logical(1), "batch_error")]
    models_sel <- intersect(input$batch_models, names(MODEL_REGISTRY))
    n_pan <- length(eps) + 1                     # +1 shared-legend panel
    ncol  <- ceiling(sqrt(n_pan))
    nrow  <- ceiling(n_pan / ncol)
    op <- par(no.readonly = TRUE); on.exit(par(op))
    par(mfrow = c(nrow, ncol),
        mar = c(2.6, 2.6, 1.1, 0.4), oma = c(0.4, 0.4, 0.2, 0.2),
        mgp = c(1.7, 0.55, 0), pty = "s")
    cols <- stats::setNames(grDevices::hcl.colors(length(models_sel),
                                                  palette = "Dark 3"),
                            models_sel)
    for (i in seq_along(eps)) {
      ep <- eps[i]
      plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i",
           xlab = "1 - Specificity", ylab = "Sensitivity",
           cex.lab = 0.85, cex.axis = 0.75)
      abline(a = 0, b = 1, lty = 3, col = "grey70")
      lg <- character(0); lc <- character(0)
      for (m in models_sel) {
        r <- br[[ep]]$roc[[m]]
        if (is.null(r)) next
        lines(1 - r$specificities, r$sensitivities, col = cols[m], lwd = 1.5)
        lg <- c(lg, sprintf("%s %.2f", MODEL_SHORT[m],
                            as.numeric(pROC::auc(r))))
        lc <- c(lc, cols[m])
      }
      mtext(paste0("(", letters[i], ") ", ep), side = 3, line = 0.1,
            adj = 0.02, font = 2, cex = 0.75)
      if (length(lg) > 0)
        legend("bottomright", legend = lg, col = lc, lwd = 1.5,
               cex = 0.52, bty = "n", inset = 0.01)
    }
    plot.new()
    legend("center", legend = paste0(models_sel, "  (", MODEL_SHORT[models_sel], ")"),
           col = cols[models_sel], lwd = 2, cex = 0.85, bty = "n",
           title = "Model", title.adj = 0.5)
  }

  output$batch_roc_plot <- renderPlot({ draw_batch_roc() })
  output$dl_batch_roc <- downloadHandler(
    filename = function() paste0("roc_all_endpoints_600dpi_", file_tag(), "_",
                                 Sys.Date(), ".png"),
    content  = function(file) save_base_600dpi(file, draw_batch_roc, 9.5, 8.5)
  )

  # Test ROC-AUC heatmap (endpoints × models): the single summary figure a
  # reviewer can scan in seconds. Title-free.
  batch_heat_ggplot <- function() {
    bm <- batch_metrics_long()
    te <- bm[bm$Set == "Test", , drop = FALSE]
    validate(need(nrow(te) > 0, "No test metrics available."))
    te$Model <- factor(te$Model, levels = rev(names(MODEL_REGISTRY)))
    te$Endpoint <- factor(te$Endpoint, levels = names(batch_results()))
    ggplot(te, aes(x = Endpoint, y = Model, fill = ROC_AUC)) +
      geom_tile(color = "white", linewidth = 0.4) +
      geom_text(aes(label = sprintf("%.2f", ROC_AUC)), size = 3) +
      scale_fill_gradient(low = "#deebf7", high = "#08306b",
                          limits = c(0.5, 1), oob = scales::squish) +
      labs(x = NULL, y = NULL, fill = "Test ROC-AUC") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 35, hjust = 1),
            panel.grid = element_blank())
  }

  output$batch_heat_plot <- renderPlot({ print(batch_heat_ggplot()) })
  output$dl_batch_heat <- downloadHandler(
    filename = function() paste0("auc_heatmap_all_endpoints_600dpi_",
                                 file_tag(), "_", Sys.Date(), ".png"),
    content = function(file)
      ggplot2::ggsave(file, batch_heat_ggplot(), width = 8, height = 5.5,
                      dpi = 600, device = "png")
  )

  # ---- About & Documentation ---------------------------------------------------
  output$about_ui <- renderUI({
    pkg_names <- c("shiny","shinydashboard","caret","pROC","dplyr","tidyr",
                   "ggplot2","randomForest","xgboost","gbm","e1071","rpart",
                   "rpart.plot","naivebayes","nnet","MASS","readxl","readr",
                   "stringr","themis","ROSE","DT")
    versions <- vapply(pkg_names, function(p) {
      tryCatch(as.character(utils::packageVersion(p)),
               error = function(e) "not installed")
    }, character(1))

    version_rows <- paste(
      sprintf("<tr><td>%s</td><td>%s</td></tr>", names(versions), versions),
      collapse = "")

    HTML(sprintf('
      <h3>Project Aim</h3>
      <p>The <b>calcaneus</b> is the largest and most frequently fractured of the
      tarsal bones, and its morphometric characteristics guide the management of
      talocalcaneal arthritis, intra-articular fractures, congenital deformities,
      pes planus, valgus malalignment and Haglund deformity. This dashboard
      provides a reproducible, publication-grade machine-learning workflow that
      classifies calcaneal clinical findings from morphometric measurements.</p>

      <h3>Scope of Calcaneal Morphometry</h3>
      <ul>
        <li><b>GA (Gissane angle):</b> critical cortical angle of the calcaneus
            (right/left), typically 95–145&deg;.</li>
        <li><b>BA (B&ouml;hler angle):</b> tuber-joint angle (right/left),
            typically 20–45&deg;; values below ~20&deg; suggest compression
            fracture.</li>
        <li><b>HD (Haglund deformity), HS (heel spur), AT (Achilles tendon
            pathology):</b> binary clinical findings recorded bilaterally.</li>
        <li><b>Demographics:</b> sex and age.</li>
      </ul>

      <h3>Methodological Gold Standards Implemented</h3>
      <ol>
        <li><b>Automatic label standardization</b> of Turkish/English tokens
            (VAR/YOK, YES/NO, K/E, F/M) with typo correction and clinical
            plausibility filters.</li>
        <li><b>Leakage-free pre-processing:</b> K-NN imputation is fitted on the
            training partition only and merely applied to the held-out test set;
            centering and scaling are fitted inside each cross-validation
            resample of the caret pipeline. (caret executes sampling before
            pre-processing within resamples, and SMOTE/ROSE cannot operate on
            missing values — hence imputation precedes resampling without ever
            touching test information.)</li>
        <li><b>Ten classifiers</b> — logistic regression, random forest,
            XGBoost, GBM, SVM (RBF), k-NN, CART, naive Bayes, neural network and
            LDA — tuned under stratified <b>5-fold cross-validation</b>.
            (XGBoost is trained through the xgboost engine directly, because
            the caret <i>xgbTree</i> wrapper is incompatible with
            xgboost &ge; 3.x; the cross-validation design, tuning metric and
            in-fold balancing are identical.)</li>
        <li><b>Class-imbalance handling</b> selectable from the interface:
            SMOTE, ROSE, up- or down-sampling, applied within resamples.</li>
        <li><b>Train vs. test evaluation</b>: Accuracy, F1, Sensitivity,
            Specificity and ROC-AUC, downloadable as CSV.</li>
        <li><b>Publication-quality graphics</b>: ROC curves, train/test bar
            charts, feature importance and white-box CART trees exportable at
            <b>600 DPI</b>. Figures are deliberately <b>title-free</b> (captions
            belong in the manuscript) and use tight, canvas-filling margins.</li>
        <li><b>Multi-endpoint batch module</b>: runs the identical pipeline
            across all binary endpoints in one click and merges them into
            <b>composite figures</b> — a multi-panel ROC figure and an
            endpoint &times; model AUC heatmap — plus a consolidated metrics
            CSV, so every endpoint fits within journal figure-count limits.</li>
        <li><b>New-case prediction:</b> enter the clinical measurements of an
            unseen case and obtain the predicted class and probability from
            every trained model; empty fields are imputed with the
            training-fitted K-NN imputer. Held-out test-set predictions are
            downloadable as CSV.</li>
        <li><b>Reproducible file names:</b> every download embeds the target,
            train/test proportion, tuneLength and seed, so each figure and
            table can be traced back to the exact settings that produced
            it.</li>
      </ol>

      <h3>Software Environment</h3>
      <p>R version: %s</p>
      <table border="1" cellpadding="5" cellspacing="0"
             style="border-collapse:collapse;">
        <tr><th>Package</th><th>Version</th></tr>
        %s
      </table>

      <h3>Suggested Citation</h3>
      <p>[Author(s)] (2026). Calcaneus Morphometry ML Dashboard: a reproducible
      Shiny pipeline for classification of calcaneal clinical findings.
      [Journal / Institution].</p>
    ', R.version.string, version_rows))
  })
}

# ============================================================================
# 6. Run
# ============================================================================
shinyApp(ui = ui, server = server)
