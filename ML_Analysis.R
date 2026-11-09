################################################################################
# MACHINE LEARNING REGRESSION ANALYSIS WITH RIGOROUS CROSS-VALIDATION
# 
# Description: RF and GLMNet regression with nested cross-validation
# Author: Meisheng Chi
# Date: December 2025
# Methods: 5-fold cross-validation repeated 10 times, with 5 independent experiments
# Models: RF (RF) and GLMNet (GLMNet) regression
################################################################################

# ENVIRONMENT SETUP ============================================================

# Clear workspace
rm(list = ls())

# Load required packages
required_packages <- c(
  "glmnet",        # GLMNet regression
  "glmnetUtils",   # Enhanced glmnet utilities
  "tidyverse",     # Data manipulation and visualization
  "recipes",       # Feature engineering
  "caret",         # Machine learning framework
  "randomForest",  # RF algorithm
  "ggsci",         # Scientific journal color palettes
  "groupdata2",    # Grouped cross-validation
  "pROC",          # ROC curves
  "gridExtra",     # Multiple plots
  "scales"         # Scale functions
)

# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load all packages
invisible(lapply(required_packages, library, character.only = TRUE))

# Set global options for reproducibility
options(stringsAsFactors = FALSE)
set.seed(42)  # Master seed for initial setup

# DATA LOADING AND PREPROCESSING ==============================================

# Load dataset
cat("\n=== DATA LOADING ===\n")
data <- read.csv(file.choose())
cat("Dataset dimensions:", dim(data)[1], "samples,", dim(data)[2], "features\n")

# Check for missing values
if(any(is.na(data))) {
  cat("WARNING: Missing values detected. Removing rows with missing values...\n")
  data <- na.omit(data)
  cat("New dataset dimensions:", dim(data)[1], "samples,", dim(data)[2], "features\n")
}

# Convert categorical variables (columns 2-21: bacterial strains) to factors
cat("Converting bacterial strain columns (2-21) to categorical factors...\n")
for (i in 2:21) {
  data[[i]] <- factor(data[[i]])
}

cat("Data preprocessing complete.\n")

# EXPERIMENTAL DESIGN PARAMETERS ===============================================

config <- list(
  n_folds = 5,
  n_cv_repeats = 10,
  n_experiments = 5,
  train_proportion = 0.75,
  test_proportion = 0.25,
  rf_ntree = 500,
  rf_mtry = 10,
  glmnet_alpha_seq = seq(0.05, 0.95, length.out = 10),
  group_variable = NULL,
  random_seeds = c(100, 234, 567, 891, 1234)
)

stopifnot(
  config$n_experiments == length(config$random_seeds),
  config$n_folds >= 2,
  config$n_cv_repeats >= 1,
  config$train_proportion > 0 & config$train_proportion < 1
)

cat("\n=== EXPERIMENTAL CONFIGURATION ===\n")
cat("Number of experiments:", config$n_experiments, "\n")
cat("Cross-validation strategy:", config$n_folds, "-fold CV repeated", 
    config$n_cv_repeats, "times\n")
cat("Train/test split:", paste0(config$train_proportion*100, "%/", 
                                config$test_proportion*100, "%\n"))

# DEFINE PUBLICATION THEME (ARIAL FONT) ========================================

theme_publication <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = "Arial", color = "black"),
      axis.title = element_text(family = "Arial", face = "bold", size = base_size),
      axis.text = element_text(family = "Arial", size = base_size - 2, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.line = element_line(color = "black", linewidth = 0.5),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position = "right",
      legend.title = element_text(family = "Arial", face = "bold", size = base_size - 2),
      legend.text = element_text(family = "Arial", size = base_size - 2),
      legend.key.size = unit(1, "lines"),
      plot.title = element_text(family = "Arial", hjust = 0.5, face = "bold", size = base_size + 2),
      plot.subtitle = element_text(family = "Arial", hjust = 0.5, size = base_size),
      plot.margin = unit(c(10, 10, 10, 10), "pt"),
      strip.text = element_text(family = "Arial", face = "bold", size = base_size)
    )
}

color_palette <- c(
  No_Model = "#999999",
  GLMNet = "#E69F00",
  RF = "#56B4E9",
  CV = "#009E73",
  Test = "#CC79A7"
)

# INITIALIZE RESULT STORAGE ====================================================

results_rmse <- data.frame()
results_classification <- data.frame()
cv_results_detailed <- list()
all_predictions <- list()

final_glmnet_for_importance <- NULL
final_rf_for_importance <- NULL
final_train_x_matrix <- NULL
final_train_data <- NULL
final_train_y <- NULL
final_preprocessing_recipe <- NULL
all_glmnet_models <- list()

# MAIN EXPERIMENTAL LOOP =======================================================

cat("\n", rep("=", 80), "\n")
cat("STARTING EXPERIMENTAL PIPELINE\n")
cat(rep("=", 80), "\n\n")

for(experiment_id in 1:config$n_experiments) {
  
  cat("\n", rep("-", 80), "\n")
  cat("EXPERIMENT", experiment_id, "OF", config$n_experiments, "\n")
  cat("Random seed:", config$random_seeds[experiment_id], "\n")
  cat(rep("-", 80), "\n\n")
  
  set.seed(config$random_seeds[experiment_id])
  
  # STEP 1: TRAIN-TEST SPLIT ---------------------------------------------------
  cat("Step 1: Creating train-test split...\n")
  
  train_indices_numeric <- createDataPartition(
    y = data$Germination_Rate, 
    p = config$train_proportion, 
    list = FALSE
  )[, 1]
  
  train_indices <- rep(FALSE, nrow(data))
  train_indices[train_indices_numeric] <- TRUE
  
  train_data <- data[train_indices, ]
  test_data <- data[!train_indices, ]
  
  cat("  Training samples:", nrow(train_data), "\n")
  cat("  Testing samples:", nrow(test_data), "\n")
  
  if(nrow(test_data) == 0) {
    stop("ERROR: Test set is empty! Check data splitting logic.")
  }
  
  classification_threshold <- median(train_data$Germination_Rate)
  cat("  Classification threshold (median):", 
      round(classification_threshold, 3), "\n\n")
  
  # STEP 2: FEATURE ENGINEERING ------------------------------------------------
  cat("Step 2: Feature engineering and preprocessing...\n")
  
  preprocessing_recipe <- recipe(Germination_Rate ~ ., data = train_data) %>%
    step_rm(matches("^Experiment$")) %>%
    step_dummy(all_nominal_predictors()) %>%
    prep()
  
  train_x <- bake(preprocessing_recipe, new_data = NULL) %>% 
    select(-Germination_Rate) %>% 
    as.matrix()
  
  test_x <- bake(preprocessing_recipe, new_data = test_data) %>%
    select(-Germination_Rate) %>%
    as.matrix()
  
  train_y <- train_data$Germination_Rate
  test_y <- test_data$Germination_Rate
  
  cat("  Feature matrix dimensions:", dim(train_x)[1], "x", dim(train_x)[2], "\n\n")
  
  # STEP 3: GLMNet MODEL WITH NESTED CV ----------------------------------------
  cat("Step 3: Training GLMNet model with nested cross-validation...\n")
  
  glmnet_cv_rmse <- numeric()
  
  for(cv_iteration in 1:config$n_cv_repeats) {
    cat("  CV iteration", cv_iteration, "of", config$n_cv_repeats, "...")
    
    set.seed(config$random_seeds[experiment_id] * 100 + cv_iteration)
    
    cv_folds <- createFolds(train_y, k = config$n_folds)
    
    if(is.list(cv_folds)) {
      fold_id_vector <- rep(NA, nrow(train_data))
      for(fold_idx in seq_along(cv_folds)) {
        fold_id_vector[cv_folds[[fold_idx]]] <- fold_idx
      }
      cv_folds <- fold_id_vector
    }
    
    cv_glmnet_model <- cva.glmnet(
      x = train_x,
      y = train_y,
      alpha = config$glmnet_alpha_seq,
      nfolds = config$n_folds,
      foldid = cv_folds,
      standardize = TRUE,
      type.measure = "mse"
    )
    
    best_alpha_idx <- which.min(sapply(cv_glmnet_model$modlist, 
                                       function(m) min(m$cvm)))
    best_cv_mse <- min(cv_glmnet_model$modlist[[best_alpha_idx]]$cvm)
    glmnet_cv_rmse <- c(glmnet_cv_rmse, sqrt(best_cv_mse))
    
    cat(" RMSE:", round(sqrt(best_cv_mse), 4), "\n")
  }
  
  cat("  Training final GLMNet model...\n")
  final_cv_glmnet <- cva.glmnet(
    x = train_x,
    y = train_y,
    alpha = config$glmnet_alpha_seq,
    standardize = TRUE,
    type.measure = "mse"
  )
  
  best_alpha_final <- which.min(sapply(final_cv_glmnet$modlist, 
                                       function(m) min(m$cvm)))
  best_glmnet_model <- final_cv_glmnet$modlist[[best_alpha_final]]
  
  test_pred_glmnet <- predict(
    best_glmnet_model, 
    newx = test_x, 
    s = "lambda.min"
  )[, 1]
  
  glmnet_test_rmse <- sqrt(mean((test_pred_glmnet - test_y)^2))
  
  cat("  GLMNet CV RMSE: mean =", round(mean(glmnet_cv_rmse), 4),
      "± SD =", round(sd(glmnet_cv_rmse), 4), "\n")
  cat("  GLMNet Test RMSE:", round(glmnet_test_rmse, 4), "\n\n")
  
  # STEP 4: RF MODEL WITH NESTED CV --------------------------------------------
  cat("Step 4: Training RF model with nested cross-validation...\n")
  
  rf_cv_rmse <- numeric()
  
  for(cv_iteration in 1:config$n_cv_repeats) {
    cat("  CV iteration", cv_iteration, "of", config$n_cv_repeats, "...")
    
    set.seed(config$random_seeds[experiment_id] * 100 + cv_iteration)
    
    cv_folds <- createFolds(train_y, k = config$n_folds)
    
    fold_rmse <- numeric(config$n_folds)
    for(fold_idx in 1:config$n_folds) {
      if(is.list(cv_folds)) {
        val_indices <- cv_folds[[fold_idx]]
      } else {
        val_indices <- which(cv_folds == fold_idx)
      }
      
      fold_train <- train_data[-val_indices, ]
      fold_val <- train_data[val_indices, ]
      
      rf_fold_model <- randomForest(
        Germination_Rate ~ . - Experiment,
        data = fold_train,
        ntree = config$rf_ntree,
        mtry = min(config$rf_mtry, ncol(fold_train) - 2),
        importance = FALSE
      )
      
      fold_predictions <- predict(rf_fold_model, newdata = fold_val)
      fold_rmse[fold_idx] <- sqrt(mean((fold_predictions - fold_val$Germination_Rate)^2))
    }
    
    rf_cv_rmse <- c(rf_cv_rmse, mean(fold_rmse))
    cat(" RMSE:", round(mean(fold_rmse), 4), "\n")
  }
  
  cat("  Training final RF model...\n")
  final_rf_model <- randomForest(
    Germination_Rate ~ . - Experiment,
    data = train_data,
    ntree = config$rf_ntree,
    mtry = min(config$rf_mtry, ncol(train_data) - 2),
    importance = TRUE,
    keep.forest = TRUE
  )
  
  test_pred_rf <- predict(final_rf_model, newdata = test_data)
  rf_test_rmse <- sqrt(mean((test_pred_rf - test_y)^2))
  
  cat("  RF CV RMSE: mean =", round(mean(rf_cv_rmse), 4),
      "± SD =", round(sd(rf_cv_rmse), 4), "\n")
  cat("  RF Test RMSE:", round(rf_test_rmse, 4), "\n\n")
  
  # STEP 5: No_Model MODEL (MEAN PREDICTION) -----------------------------------
  cat("Step 5: Evaluating No_Model model...\n")
  
  No_Model_prediction <- rep(mean(train_y), length(test_y))
  No_Model_rmse <- sqrt(mean((No_Model_prediction - test_y)^2))
  
  cat("  No_Model (mean) RMSE:", round(No_Model_rmse, 4), "\n\n")
  
  # STEP 6: STORE RMSE RESULTS -------------------------------------------------
  results_rmse <- rbind(
    results_rmse,
    data.frame(
      Experiment = experiment_id,
      Seed = config$random_seeds[experiment_id],
      No_Model_RMSE = No_Model_rmse,
      GLMNet_CV_Mean = mean(glmnet_cv_rmse),
      GLMNet_CV_SD = sd(glmnet_cv_rmse),
      GLMNet_Test_RMSE = glmnet_test_rmse,
      RF_CV_Mean = mean(rf_cv_rmse),
      RF_CV_SD = sd(rf_cv_rmse),
      RF_Test_RMSE = rf_test_rmse
    )
  )
  
  cv_results_detailed[[experiment_id]] <- list(
    glmnet_cv = glmnet_cv_rmse,
    rf_cv = rf_cv_rmse
  )
  
  all_predictions[[experiment_id]] <- list(
    test_y = test_y,
    threshold = classification_threshold,
    glmnet_pred = test_pred_glmnet,
    rf_pred = test_pred_rf,
    No_Model_pred = No_Model_prediction
  )
  
  # STEP 7: CLASSIFICATION METRICS ---------------------------------------------
  cat("Step 7: Computing binary classification metrics...\n")
  
  test_class_true <- factor(
    ifelse(test_y >= classification_threshold, "High", "Low"),
    levels = c("High", "Low")
  )
  
  calculate_classification_metrics <- function(predictions, true_labels) {
    pred_classes <- factor(
      ifelse(predictions >= classification_threshold, "High", "Low"),
      levels = c("High", "Low")
    )
    
    confusion_matrix <- table(Predicted = pred_classes, Actual = true_labels)
    
    total <- sum(confusion_matrix)
    
    tp <- ifelse("High" %in% rownames(confusion_matrix) && "High" %in% colnames(confusion_matrix),
                 confusion_matrix["High", "High"], 0)
    fp <- ifelse("High" %in% rownames(confusion_matrix) && "Low" %in% colnames(confusion_matrix),
                 confusion_matrix["High", "Low"], 0)
    tn <- ifelse("Low" %in% rownames(confusion_matrix) && "Low" %in% colnames(confusion_matrix),
                 confusion_matrix["Low", "Low"], 0)
    fn <- ifelse("Low" %in% rownames(confusion_matrix) && "High" %in% colnames(confusion_matrix),
                 confusion_matrix["Low", "High"], 0)
    
    data.frame(
      True_Positive = tp / total,
      False_Positive = fp / total,
      True_Negative = tn / total,
      False_Negative = fn / total
    )
  }
  
  No_Model_metrics <- calculate_classification_metrics(No_Model_prediction, test_class_true)
  glmnet_metrics <- calculate_classification_metrics(test_pred_glmnet, test_class_true)
  rf_metrics <- calculate_classification_metrics(test_pred_rf, test_class_true)
  
  results_classification <- rbind(
    results_classification,
    cbind(Model = "No_Model", Experiment = experiment_id, 
          Seed = config$random_seeds[experiment_id], No_Model_metrics),
    cbind(Model = "GLMNet", Experiment = experiment_id, 
          Seed = config$random_seeds[experiment_id], glmnet_metrics),
    cbind(Model = "RF", Experiment = experiment_id, 
          Seed = config$random_seeds[experiment_id], rf_metrics)
  )
  
  cat("  Classification metrics computed.\n")
  
  # STEP 8: SAVE MODELS FOR FEATURE IMPORTANCE ---------------------------------
  if(experiment_id == config$n_experiments) {
    final_glmnet_for_importance <- best_glmnet_model
    final_rf_for_importance <- final_rf_model
    final_train_x_matrix <- train_x
    final_train_data <- train_data
    final_train_y <- train_y
    final_preprocessing_recipe <- preprocessing_recipe
    all_glmnet_models <- final_cv_glmnet
  }
  
  cat("\nExperiment", experiment_id, "completed.\n")
}

# RESULTS SUMMARY ==============================================================

cat("\n", rep("=", 80), "\n")
cat("RESULTS SUMMARY ACROSS ALL EXPERIMENTS\n")
cat(rep("=", 80), "\n\n")

rmse_summary <- results_rmse %>%
  summarise(
    No_Model_Mean = mean(No_Model_RMSE),
    No_Model_SD = sd(No_Model_RMSE),
    GLMNet_Mean = mean(GLMNet_Test_RMSE),
    GLMNet_SD = sd(GLMNet_Test_RMSE),
    GLMNet_CV_Avg_Mean = mean(GLMNet_CV_Mean),
    GLMNet_CV_Avg_SD = mean(GLMNet_CV_SD),
    RF_Mean = mean(RF_Test_RMSE),
    RF_SD = sd(RF_Test_RMSE),
    RF_CV_Avg_Mean = mean(RF_CV_Mean),
    RF_CV_Avg_SD = mean(RF_CV_SD)
  )

print(rmse_summary)

cat("\n--- Exporting Results ---\n")
write_csv(results_rmse, "Results_RMSE_Detailed.csv")
write_csv(results_classification, "Results_Classification_Detailed.csv")
cat("Results saved to CSV files.\n")

# ==================================================================================
# PUBLICATION-QUALITY VISUALIZATIONS (ALL FONTS: ARIAL)
# ==================================================================================

cat("\n", rep("=", 80), "\n")
cat("GENERATING PUBLICATION-QUALITY FIGURES\n")
cat(rep("=", 80), "\n\n")

# ==================================================================================
# FIGURE 1: RMSE COMPARISON (CV vs Test)
# ==================================================================================
cat("Creating Figure 1: RMSE comparison...\n")

fig1_data <- results_rmse %>%
  select(Experiment, GLMNet_CV_Mean, GLMNet_Test_RMSE, 
         RF_CV_Mean, RF_Test_RMSE) %>%
  pivot_longer(
    cols = -Experiment, 
    names_to = "Metric", 
    values_to = "RMSE"
  ) %>%
  separate(Metric, into = c("Model", "Type"), sep = "_(?=(CV|Test))") %>%
  mutate(
    Type = gsub("_.*", "", Type),
    Type = factor(Type, levels = c("CV", "Test")),
    Model = factor(Model, levels = c("GLMNet", "RF"))
  )

figure1 <- ggplot(fig1_data, aes(x = Model, y = RMSE, fill = Type)) +
  geom_boxplot(
    position = position_dodge(0.8), 
    width = 0.6, 
    alpha = 0.8,
    outlier.shape = NA,
    linewidth = 0.8
  ) +
  geom_point(
    aes(color = Type),
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
    size = 3,
    alpha = 0.7,
    shape = 21,
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c(CV = "#6DADF9", Test = "#F6B293"),
    labels = c("Cross-Validation", "Test Set")
  ) +
  scale_color_manual(
    values = c(CV = "#6DADF9", Test = "#F6B293")
  ) +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = "Root Mean Squared Error (RMSE)",
    fill = "Evaluation"
  ) +
  theme_publication()

ggsave("Figure1_RMSE_Comparison.jpg", figure1, width = 8, height = 6, dpi = 600)
ggsave("Figure1_RMSE_Comparison.svg", figure1, width = 8, height = 6)
cat("  Saved: Figure1_RMSE_Comparison.jpg/svg\n")

# ==================================================================================
# FIGURE 2: TEST SET RMSE BAR CHART
# ==================================================================================
cat("Creating Figure 2: Test Set RMSE comparison...\n")

# Load required packages
library(tidyverse)
library(agricolae)  # For Tukey HSD test and letter grouping

# Data preparation
fig2_data <- results_rmse %>%
  select(Experiment, No_Model_RMSE, GLMNet_Test_RMSE, RF_Test_RMSE) %>%
  pivot_longer(
    cols = -Experiment,
    names_to = "Model",
    values_to = "RMSE"
  ) %>%
  mutate(
    Model = factor(
      gsub("_.*", "", Model),
      levels = c("No", "GLMNet", "RF"),
      labels = c("No_model", "GLMNet", "RF")
    )
  )

# Summary statistics
fig2_summary <- fig2_data %>%
  group_by(Model) %>%
  summarise(
    Mean = mean(RMSE),
    SD = sd(RMSE),
    SE = SD / sqrt(n())
  )

# ============================================
# Significance analysis: ANOVA + Tukey HSD
# ============================================
# Perform one-way ANOVA
anova_result <- aov(RMSE ~ Model, data = fig2_data)
summary(anova_result)

# Tukey HSD test with letter grouping
tukey_result <- HSD.test(anova_result, "Model", group = TRUE)
print(tukey_result)

# Extract letter labels and merge with summary data
sig_letters <- tukey_result$groups %>%
  rownames_to_column("Model") %>%
  select(Model, groups) %>%
  rename(Letter = groups)

# Merge significance letters with summary data
fig2_summary <- fig2_summary %>%
  left_join(sig_letters, by = "Model") %>%
  mutate(
    # Calculate label position (above error bar)
    label_y = Mean + SE + max(Mean) * 0.03
  )

# ============================================
# Optimized plotting code with significance letters
# ============================================
figure2 <- ggplot(fig2_summary, aes(x = Model, y = Mean, fill = Model)) +
  # Bar plot - adjust width and transparency
  geom_bar(stat = "identity", width = 0.45, color = "black", linewidth = 0.3) +
  
  # Error bars - black color with appropriate thickness
  geom_errorbar(
    aes(ymin = Mean - SE, ymax = Mean + SE),
    width = 0.30,
    linewidth = 0.6,
    color = "black"
  ) +
  
  # Scatter points - adjust size and transparency
  geom_point(
    data = fig2_data,
    aes(x = Model, y = RMSE),
    position = position_jitter(width = 0.08, seed = 123),
    size = 2,
    alpha = 0.6,
    shape = 21,
    fill = "grey30",
    color = "black",
    stroke = 0.3
  ) +
  
  # Significance letters
  geom_text(
    aes(x = Model, y = label_y, label = Letter),
    size = 8,
    fontface = "bold",
    color = "black"
  ) +
  
  # Color scheme
  scale_fill_manual(values = c(
    "No_model" = "#999999", 
    "GLMNet" = "#CB5623", 
    "RF" = "#45ABBB"
  )) +
  
  # y-axis starts from 0, with extra space at top for letters
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.1)),  # 10% space at top for letters
    limits = c(0, 0.15)
  ) +
  
  # Labels
  labs(
    title = NULL,
    x = "Model",
    y = "RMSE"
  ) +
  
  theme_classic(base_size = 8, base_family = "Arial") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.8),
    axis.line = element_blank(),
    axis.ticks = element_line(linewidth = 1.2, color = "black"),
    axis.ticks.length = unit(1.5, "mm"),
    axis.title = element_text(size = 20, face = "bold", color = "black"),
    axis.text = element_text(size = 18, color = "black"),
    legend.position = "none",
    panel.grid = element_blank(),
    panel.background = element_blank()
  )

ggsave("Figure2_Test_RMSE.jpg", figure2, width = 8, height = 6, dpi = 600)
ggsave("Figure2_Test_RMSE.svg", figure2, width = 8, height = 6)
cat("  Saved: Figure2_Test_RMSE.jpg/svg\n")

# ==================================================================================
# FIGURE 3: ROC CURVES (CORRECTED)
# ==================================================================================
cat("Creating Figure 3: ROC curves...\n")

last_exp <- all_predictions[[config$n_experiments]]
threshold <- last_exp$threshold

true_labels <- ifelse(last_exp$test_y >= threshold, 1, 0)

roc_glmnet <- roc(true_labels, last_exp$glmnet_pred, quiet = TRUE)
roc_rf <- roc(true_labels, last_exp$rf_pred, quiet = TRUE)

roc_data_glmnet <- data.frame(
  Model = "GLMNet",
  FPR = 1 - roc_glmnet$specificities,
  TPR = roc_glmnet$sensitivities
)

roc_data_rf <- data.frame(
  Model = "RF",
  FPR = 1 - roc_rf$specificities,
  TPR = roc_rf$sensitivities
)

roc_data <- rbind(roc_data_glmnet, roc_data_rf)

auc_glmnet <- round(auc(roc_glmnet), 3)
auc_rf <- round(auc(roc_rf), 3)

figure3 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("GLMNet" = "#E69F00", "RF" = "#56B4E9"),
    labels = c(
      paste0("GLMNet (AUC = ", auc_glmnet, ")"),
      paste0("RF (AUC = ", auc_rf, ")")
    )
  ) +
  labs(
    title = "ROC Curves",
    subtitle = "Binary classification of germination rate",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    color = "Model"
  ) +
  coord_equal() +
  theme_publication() +
  theme(
    legend.position = c(0.7, 0.25),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.5)
  )

ggsave("Figure3_ROC_Curves.jpg", figure3, width = 8, height = 8, dpi = 600)
ggsave("Figure3_ROC_Curves.svg", figure3, width = 8, height = 8)
cat("  Saved: Figure3_ROC_Curves.jpg/svg\n")

# ==================================================================================
# FIGURE 4: GLMNET COEFFICIENT TRAJECTORY (REGULARIZATION PATH)
# ==================================================================================
cat("Creating Figure 4: GLMNet coefficient trajectory...\n")

best_alpha <- config$glmnet_alpha_seq[
  which.min(sapply(all_glmnet_models$modlist, function(m) min(m$cvm)))
]

glmnet_path <- glmnet(
  x = final_train_x_matrix,
  y = final_train_y,
  alpha = best_alpha,
  standardize = TRUE
)

coef_matrix <- as.matrix(glmnet_path$beta)
lambda_values <- glmnet_path$lambda

strain_feature_names <- colnames(data)[2:21]

coef_aggregated <- matrix(0, nrow = length(strain_feature_names), ncol = length(lambda_values))
rownames(coef_aggregated) <- strain_feature_names

for(i in seq_along(strain_feature_names)) {
  feature <- strain_feature_names[i]
  matching_cols <- grep(paste0("^", feature, "(_|$)"), rownames(coef_matrix))
  if(length(matching_cols) > 0) {
    coef_aggregated[i, ] <- colSums(abs(coef_matrix[matching_cols, , drop = FALSE]))
  }
}

coef_long <- as.data.frame(coef_aggregated) %>%
  mutate(Feature = rownames(coef_aggregated)) %>%
  pivot_longer(
    cols = -Feature,
    names_to = "Lambda_Index",
    values_to = "Coefficient"
  ) %>%
  mutate(
    Lambda_Index = as.numeric(gsub("V", "", Lambda_Index)),
    Log_Lambda = log(lambda_values[Lambda_Index])
  )

top_features <- coef_long %>%
  filter(Lambda_Index == which.min(lambda_values)) %>%
  arrange(desc(abs(Coefficient))) %>%
  head(10) %>%
  pull(Feature)

coef_plot_data <- coef_long %>%
  filter(Feature %in% top_features)

figure4 <- ggplot(coef_plot_data, aes(x = Log_Lambda, y = Coefficient, color = Feature)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = log(final_glmnet_for_importance$lambda.min),
    linetype = "dashed",
    color = "red",
    linewidth = 0.8
  ) +
  annotate(
    "text",
    x = log(final_glmnet_for_importance$lambda.min) + 0.3,
    y = max(coef_plot_data$Coefficient) * 0.9,
    label = expression(lambda[min]),
    family = "Arial",
    size = 4
  ) +
  scale_color_brewer(palette = "Paired") +
  labs(
    title = "GLMNet Coefficient Trajectory",
    subtitle = paste0("Regularization path (α = ", round(best_alpha, 2), ")"),
    x = expression(log(lambda)),
    y = "|Coefficient|",
    color = "Bacterial Strain"
  ) +
  theme_publication() +
  theme(
    legend.position = "right",
    legend.key.height = unit(0.8, "lines")
  )

ggsave("Figure4_Coefficient_Trajectory.jpg", figure4, width = 12, height = 8, dpi = 600)
ggsave("Figure4_Coefficient_Trajectory.svg", figure4, width = 12, height = 8)
cat("  Saved: Figure4_Coefficient_Trajectory.jpg/svg\n")

# ==================================================================================
# FEATURE IMPORTANCE ANALYSIS 
# ==================================================================================
cat("\n", rep("=", 80), "\n")
cat("FEATURE IMPORTANCE ANALYSIS (UNIFIED PERMUTATION IMPORTANCE)\n")
cat(rep("=", 80), "\n\n")

strain_feature_names <- colnames(data)[2:21]
cat("Analyzing importance of", length(strain_feature_names), "bacterial strains...\n")
cat("Using unified permutation importance method for both models...\n\n")

# 计算RF的Permutation Importance -----------------------------------------------
cat("Calculating RF permutation importance...\n")


rf_base_pred <- predict(final_rf_for_importance, newdata = final_train_data)
rf_base_mse <- mean((rf_base_pred - final_train_y)^2)

rf_perm_importance <- sapply(strain_feature_names, function(feature) {

  permuted_data <- final_train_data
  
  set.seed(42)
  perm_idx <- sample(1:nrow(permuted_data))
  permuted_data[[feature]] <- permuted_data[[feature]][perm_idx]
  perm_pred <- predict(final_rf_for_importance, newdata = permuted_data)
  perm_mse <- mean((perm_pred - final_train_y)^2)

  return((perm_mse - rf_base_mse) / rf_base_mse * 100)
})


rf_perm_importance[rf_perm_importance < 0] <- 0
rf_importance_normalized <- (rf_perm_importance / sum(rf_perm_importance)) * 100

rf_importance_df <- data.frame(
  Feature = strain_feature_names,
  RF_Importance = as.numeric(rf_importance_normalized)
)

cat("  Top 5 features (RF):\n")
print(rf_importance_df %>% arrange(desc(RF_Importance)) %>% head(5))

# Permutation Importance -------------------------------------------
cat("\nCalculating GLMNet permutation importance...\n")

glmnet_base_pred <- predict(final_glmnet_for_importance, newx = final_train_x_matrix, s = "lambda.min")[,1]
glmnet_base_mse <- mean((glmnet_base_pred - final_train_y)^2)

glmnet_perm_importance <- sapply(strain_feature_names, function(feature) {
  matching_cols <- grep(paste0("^", feature, "(_|$)"), colnames(final_train_x_matrix))
  
  if(length(matching_cols) == 0) return(0)
  
  permuted_x <- final_train_x_matrix
  
  set.seed(42)
  perm_idx <- sample(1:nrow(permuted_x))
  permuted_x[, matching_cols] <- permuted_x[perm_idx, matching_cols]
  
  perm_pred <- predict(final_glmnet_for_importance, newx = permuted_x, s = "lambda.min")[,1]
  perm_mse <- mean((perm_pred - final_train_y)^2)
  
  return((perm_mse - glmnet_base_mse) / glmnet_base_mse * 100)
})

glmnet_perm_importance[glmnet_perm_importance < 0] <- 0
glmnet_importance_normalized <- (glmnet_perm_importance / sum(glmnet_perm_importance)) * 100

glmnet_importance_df <- data.frame(
  Feature = strain_feature_names,
  GLMNet_Importance = as.numeric(glmnet_importance_normalized)
)

cat("  Top 5 features (GLMNet):\n")
print(glmnet_importance_df %>% arrange(desc(GLMNet_Importance)) %>% head(5))

# COMBINE IMPORTANCE SCORES ----------------------------------------------------
cat("\nCombining feature importance scores...\n")

importance_combined <- merge(rf_importance_df, glmnet_importance_df, by = "Feature")

importance_combined <- importance_combined %>%
  mutate(Avg_Importance = (RF_Importance + GLMNet_Importance) / 2) %>%
  arrange(desc(Avg_Importance))

cat("  Combined importance calculated for", nrow(importance_combined), "strains.\n")

write_csv(importance_combined, "Feature_Importance_Scores.csv")
cat("  Saved: Feature_Importance_Scores.csv\n")

# ==================================================================================
# FIGURE 5: FEATURE IMPORTANCE BAR CHART 
# ==================================================================================
cat("\nCreating Figure 5: Feature importance bar chart...\n")

fig5_data <- importance_combined %>%
  select(Feature, RF_Importance, GLMNet_Importance) %>%
  pivot_longer(
    cols = c(RF_Importance, GLMNet_Importance),
    names_to = "Model",
    values_to = "Importance"
  ) %>%
  mutate(
    Model = factor(
      Model,
      levels = c("RF_Importance", "GLMNet_Importance"),
      labels = c("RF", "GLMNet")
    ),
    Feature = factor(Feature, levels = rev(importance_combined$Feature))
  )

figure5 <- ggplot(fig5_data, aes(x = Feature, y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("RF" = "#56B4E9", "GLMNet" = "#E69F00")) +
  labs(
    title = "Feature Importance Analysis",
    subtitle = "Permutation importance of bacterial strains in predicting germination rate",
    x = "Bacterial Strain",
    y = "Importance (%)",
    fill = "Model"
  ) +
  theme_publication() +
  theme(
    legend.position = "top",
    legend.title = element_text(family = "Arial"),
    axis.text.y = element_text(family = "Arial", size = 10)
  )

ggsave("Figure5_Feature_Importance_Bar.jpg", figure5, width = 10, height = 12, dpi = 600)
ggsave("Figure5_Feature_Importance_Bar.svg", figure5, width = 10, height = 12)
cat("  Saved: Figure5_Feature_Importance_Bar.jpg/svg\n")

# ==================================================================================
# FIGURE 6: CV STABILITY ACROSS ITERATIONS
# ==================================================================================
cat("\nCreating Figure 6: CV stability across iterations...\n")

cv_stability_data <- do.call(rbind, lapply(seq_along(cv_results_detailed), function(exp_id) {
  data.frame(
    Experiment = exp_id,
    Iteration = 1:config$n_cv_repeats,
    GLMNet = cv_results_detailed[[exp_id]]$glmnet_cv,
    RF = cv_results_detailed[[exp_id]]$rf_cv
  )
})) %>%
  pivot_longer(
    cols = c(GLMNet, RF),
    names_to = "Model",
    values_to = "CV_RMSE"
  )

figure6 <- ggplot(cv_stability_data, aes(x = Iteration, y = CV_RMSE, color = Model)) +
  geom_line(alpha = 0.6, linewidth = 0.8) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~ Experiment, ncol = 5, labeller = label_both) +
  scale_color_manual(
    values = c("GLMNet" = "#E69F00", "RF" = "#56B4E9")
  ) +
  labs(
    title = "Cross-Validation Stability Across Iterations",
    subtitle = "10 CV repeats per experiment",
    x = "CV Iteration",
    y = "Cross-Validation RMSE",
    color = "Model"
  ) +
  theme_publication() +
  theme(
    strip.text = element_text(family = "Arial", face = "bold")
  )

ggsave("Figure6_CV_Stability.jpg", figure6, width = 14, height = 6, dpi = 600)
ggsave("Figure6_CV_Stability.svg", figure6, width = 14, height = 6)
cat("  Saved: Figure6_CV_Stability.jpg/svg\n")

# ==================================================================================
# FIGURE 7: CLASSIFICATION METRICS HEATMAP
# ==================================================================================
cat("\nCreating Figure 7: Classification metrics heatmap...\n")

fig7_data <- results_classification %>%
  group_by(Model) %>%
  summarise(
    TP = mean(True_Positive),
    FP = mean(False_Positive),
    TN = mean(True_Negative),
    FN = mean(False_Negative),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(TP, FP, TN, FN),
    names_to = "Metric",
    values_to = "Proportion"
  ) %>%
  mutate(
    Model = factor(Model, levels = c("No_Model", "GLMNet", "RF")),
    Metric = factor(Metric, levels = c("TP", "FP", "FN", "TN"))
  )

figure7 <- ggplot(fig7_data, aes(x = Metric, y = Model, fill = Proportion)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(
    aes(label = sprintf("%.1f%%", Proportion * 100)),
    size = 5,
    family = "Arial",
    fontface = "bold"
  ) +
  scale_fill_gradient2(
    low = "#FFFFFF",
    mid = "#FDD49E",
    high = "#D94701",
    midpoint = 0.25,
    name = "Proportion",
    labels = scales::percent
  ) +
  labs(
    title = "Classification Performance",
    subtitle = "Confusion matrix proportions (averaged across experiments)",
    x = "Metric",
    y = "Model"
  ) +
  theme_publication() +
  theme(
    panel.grid = element_blank()
  )

ggsave("Figure7_Classification_Metrics.jpg", figure7, width = 10, height = 6, dpi = 600)
ggsave("Figure7_Classification_Metrics.svg", figure7, width = 10, height = 6)
cat("  Saved: Figure7_Classification_Metrics.jpg/svg\n")

# STATISTICAL SIGNIFICANCE TESTING =============================================

cat("\n", rep("=", 80), "\n")
cat("STATISTICAL SIGNIFICANCE TESTING\n")
cat(rep("=", 80), "\n\n")

t_test_glmnet_No_Model <- t.test(
  results_rmse$GLMNet_Test_RMSE,
  results_rmse$No_Model_RMSE,
  paired = TRUE
)

t_test_rf_No_Model <- t.test(
  results_rmse$RF_Test_RMSE,
  results_rmse$No_Model_RMSE,
  paired = TRUE
)

t_test_rf_glmnet <- t.test(
  results_rmse$RF_Test_RMSE,
  results_rmse$GLMNet_Test_RMSE,
  paired = TRUE
)

cat("GLMNet vs Baseline:\n")
cat("  Mean difference:", round(t_test_glmnet_No_Model$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_glmnet_No_Model$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_glmnet_No_Model$p.value, digits = 3), "\n\n")

cat("RF vs Baseline:\n")
cat("  Mean difference:", round(t_test_rf_No_Model$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_rf_No_Model$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_rf_No_Model$p.value, digits = 3), "\n\n")

cat("RF vs GLMNet:\n")
cat("  Mean difference:", round(t_test_rf_glmnet$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_rf_glmnet$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_rf_glmnet$p.value, digits = 3), "\n\n")

statistical_tests <- data.frame(
  Comparison = c("GLMNet vs Baseline", "RF vs Baseline", "RF vs GLMNet"),
  Mean_Difference = c(
    t_test_glmnet_No_Model$estimate,
    t_test_rf_No_Model$estimate,
    t_test_rf_glmnet$estimate
  ),
  T_Statistic = c(
    t_test_glmnet_No_Model$statistic,
    t_test_rf_No_Model$statistic,
    t_test_rf_glmnet$statistic
  ),
  P_Value = c(
    t_test_glmnet_No_Model$p.value,
    t_test_rf_No_Model$p.value,
    t_test_rf_glmnet$p.value
  )
)

write_csv(statistical_tests, "Statistical_Test_Results.csv")
cat("Statistical test results saved to: Statistical_Test_Results.csv\n")

# FINAL SUMMARY ================================================================

cat("\n", rep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("Summary of outputs:\n\n")

cat("CSV files:\n")
cat("  * Results_RMSE_Detailed.csv\n")
cat("  * Results_Classification_Detailed.csv\n")
cat("  * Feature_Importance_Scores.csv\n")
cat("  * Statistical_Test_Results.csv\n\n")

cat("Figures (JPEG and SVG, Arial font):\n")
cat("  * Figure1_RMSE_Comparison\n")
cat("  * Figure2_Test_RMSE\n")
cat("  * Figure3_ROC_Curves\n")
cat("  * Figure4_Coefficient_Trajectory\n")
cat("  * Figure5_Feature_Importance_Bar (并排柱形图)\n")
cat("  * Figure6_CV_Stability\n")
cat("  * Figure7_Classification_Metrics\n\n")

cat("\n", rep("=", 80), "\n")
cat("All analyses completed successfully!\n")
cat("Results are ready for manuscript submission.\n")
cat(rep("=", 80), "\n\n")

################################################################################
# END OF SCRIPT
################################################################################
