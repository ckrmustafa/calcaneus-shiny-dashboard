# ============================================================================
# data_prep.R
# ----------------------------------------------------------------------------
# Data Cleaning & Preparation Pipeline
# Project : Calcaneus Morphometry — Machine Learning Dashboard
# Domain  : Medicine / Biomechanics (calcaneal morphometry)
# Author  : [Your Name / Research Group]
# ----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES
#   1. Reads the raw dataset (.xlsx or .csv).
#   2. Standardizes Turkish/English categorical labels automatically:
#        - Presence/absence : VAR/YOK, EVET/HAYIR, YES/NO, V/Y, E/H, 1/0 ...
#        - Sex              : K/E, F/M, KADIN/ERKEK, FEMALE/MALE ...
#        - Fixes whitespace, case, and known typos (e.g., "EK" -> Male,
#          "TOK" -> YOK).
#   3. Coerces numeric morphometric variables (Gissane angle, Boehler angle,
#      Age) and removes impossible entries.
#   4. Writes a clean, analysis-ready CSV (calcaneus_clean.csv).
#
# NOTE ON MISSING VALUES
#   Missing values are intentionally NOT imputed here. Imputation is performed
#   inside the caret training pipeline (preProcess = "knnImpute"), i.e. it is
#   re-estimated inside every cross-validation resample and fitted on the
#   training partition only. This prevents data leakage between train/test.
# ============================================================================

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("readxl", "dplyr", "stringr", "janitor", "readr")
to_install <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                    logical(1), quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install)

library(readxl)
library(dplyr)
library(stringr)
library(janitor)
library(readr)

# ---- 1. User settings ------------------------------------------------------
INPUT_FILE  <- "HAM_VERI.xlsx"          # raw data (.xlsx or .csv)
OUTPUT_FILE <- "calcaneus_clean.csv"    # cleaned output used by app.R

# ============================================================================
# 2. Standardization helpers
# ============================================================================

# Normalize any free-text label to an upper-cased, trimmed token
.norm_token <- function(x) {
  x <- as.character(x)
  x <- str_trim(x)
  x <- str_squish(x)
  toupper(x)
}

# --- Presence / absence labels (VAR/YOK, YES/NO, ...) -----------------------
# Returns factor with levels c("Yes","No"); unrecognised tokens -> NA.
standardize_binary <- function(x) {
  t <- .norm_token(x)
  yes_tokens <- c("VAR", "V", "YES", "Y", "EVET", "E+", "PRESENT", "POS",
                  "POSITIVE", "1", "TRUE", "T")
  no_tokens  <- c("YOK", "N", "NO", "HAYIR", "H", "ABSENT", "NEG",
                  "NEGATIVE", "0", "FALSE", "F-", "NONE")
  out <- dplyr::case_when(
    t %in% yes_tokens            ~ "Yes",
    t %in% no_tokens             ~ "No",
    # known typos observed in the raw export
    t %in% c("TOK")              ~ "No",
    t %in% c("VARR", "VAE")      ~ "Yes",
    t == "" | t == "NA" | is.na(t) ~ NA_character_,
    TRUE                         ~ NA_character_
  )
  factor(out, levels = c("No", "Yes"))
}

# --- Sex labels (K/E, F/M, ...) ----------------------------------------------
standardize_sex <- function(x) {
  t <- .norm_token(x)
  out <- dplyr::case_when(
    t %in% c("K", "F", "KADIN", "KIZ", "FEMALE", "WOMAN") ~ "Female",
    t %in% c("E", "M", "ERKEK", "MALE", "MAN", "EK")      ~ "Male",
    t == "" | t == "NA" | is.na(t)                        ~ NA_character_,
    TRUE                                                  ~ NA_character_
  )
  factor(out, levels = c("Female", "Male"))
}

# --- Safe numeric coercion ---------------------------------------------------
safe_numeric <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  t <- .norm_token(x)
  t <- gsub(",", ".", t)                 # decimal comma -> dot
  suppressWarnings(as.numeric(t))
}

# ============================================================================
# 3. Read the raw file
# ============================================================================
read_raw <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("xlsx", "xls")) {
    readxl::read_excel(path)
  } else if (ext == "csv") {
    readr::read_csv(path, show_col_types = FALSE)
  } else {
    stop("Unsupported file type: ", ext)
  }
}

raw <- read_raw(INPUT_FILE)
raw <- janitor::clean_names(raw)    # syntactically valid, lowercase names
message("Raw data: ", nrow(raw), " rows x ", ncol(raw), " columns")

# ============================================================================
# 4. Variable-specific cleaning
#    Column naming convention of this project:
#      Sex, Age, GA_R/GA_L (Gissane angle, right/left),
#      BA_R/BA_L (Boehler angle), HD (Haglund deformity),
#      HS (heel spur / plantar spur), AT (Achilles tendon pathology)
# ============================================================================
dat <- raw %>%
  mutate(
    # --- demographics ---
    sex = standardize_sex(sex),
    age = safe_numeric(age),
    # --- morphometric angles ---
    ga_r = safe_numeric(ga_r),
    ga_l = safe_numeric(ga_l),
    ba_r = safe_numeric(ba_r),
    ba_l = safe_numeric(ba_l),
    # --- binary clinical findings ---
    hd_r = standardize_binary(hd_r),
    hd_l = standardize_binary(hd_l),
    hs_r = standardize_binary(hs_r),
    hs_l = standardize_binary(hs_l),
    at_r = standardize_binary(at_r),
    at_l = standardize_binary(at_l)
  )

# --- Plausibility filters (clinical range checks) ---------------------------
# Gissane angle is typically ~95-145 degrees; Boehler angle ~20-45 degrees.
# Entries outside broad physiological bounds are set to NA (not deleted),
# so that K-NN imputation can handle them inside the ML pipeline.
dat <- dat %>%
  mutate(
    age  = ifelse(age  < 0 | age  > 110, NA, age),
    ga_r = ifelse(ga_r < 60 | ga_r > 170, NA, ga_r),
    ga_l = ifelse(ga_l < 60 | ga_l > 170, NA, ga_l),
    ba_r = ifelse(ba_r < 5  | ba_r > 70,  NA, ba_r),
    ba_l = ifelse(ba_l < 5  | ba_l > 70,  NA, ba_l)
  )

# --- Drop rows with no usable information -----------------------------------
n_before <- nrow(dat)
dat <- dat %>%
  filter(!is.na(sex)) %>%                       # sex is a key predictor
  filter(!if_all(everything(), is.na))
message("Rows removed (unusable): ", n_before - nrow(dat))

# ============================================================================
# 5. Cleaning report
# ============================================================================
cat("\n===== CLEANING REPORT =====\n")
cat("Final sample size :", nrow(dat), "\n")
cat("Sex distribution  :\n"); print(table(dat$sex, useNA = "ifany"))
cat("\nMissing values per variable:\n")
print(colSums(is.na(dat)))
cat("\nBinary findings (counts):\n")
for (v in c("hd_r","hd_l","hs_r","hs_l","at_r","at_l")) {
  cat("--", v, ": "); print(table(dat[[v]], useNA = "ifany"))
}

# ============================================================================
# 6. Write clean data
# ============================================================================
readr::write_csv(dat, OUTPUT_FILE)
message("\nClean dataset written to: ", normalizePath(OUTPUT_FILE))
message("Run the dashboard with: shiny::runApp('app.R')")

# ============================================================================
# 7. Methodological note (for reviewers / users)
# ============================================================================
cat("\n")
cat(strrep("=", 70), "\n")
cat("METHODOLOGICAL NOTE ON OVERFITTING\n")
cat(strrep("=", 70), "\n")
cat("
The clinical dataset (n=", nrow(dat), ") has limited sample size for ML.\n")
cat("Tree-based ensembles (RF, XGBoost) may show high Train AUC (~1.00)\n")
cat("but lower Test AUC (~0.74-0.77), indicating memorization. This is\n")
cat("EXPECTED and should be reported honestly in publications.\n\n")
cat("Anti-overfitting measures implemented in app.R:\n")
cat("  - RF: nodesize=10, mtry=2 (shallower trees)\n")
cat("  - GBM: bag.fraction=0.8 (stochastic gradient boosting)\n")
cat("  - CART: cp=0.02, minbucket=20, maxdepth=5 (pruned)\n")
cat("  - NNT: decay=0.1 (weight regularization)\n")
cat("  - XGBoost: eta=0.05-0.3, max_depth=3-6, subsample=0.8\n")
cat("  - All: stratified 5-fold CV, balancing inside folds only\n\n")
cat("For publication, report BOTH train and test metrics, and discuss\n")
cat("the Train-Test gap explicitly in the Discussion section.\n")
cat(strrep("=", 70), "\n")
