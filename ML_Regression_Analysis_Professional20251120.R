################################################################################
# MACHINE LEARNING REGRESSION ANALYSIS WITH RIGOROUS CROSS-VALIDATION
# 
# Description: RF and GLMNet regression with nested cross-validation
# Author: Meisheng Chi
# Date: November 2025
# Methods: 5-fold cross-validation repeated 10 times, with 5 independent experiments
# Models: RF (RF) and GLMNet (GLMNet) regression
#
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
  "groupdata2"     # Grouped cross-validation
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
# Column 1 is Experiment ID, columns 2-21 are bacterial strains
cat("Converting bacterial strain columns (2-21) to categorical factors...\n")
for (i in 2:21) {
  data[[i]] <- factor(data[[i]])
}

cat("Data preprocessing complete.\n")

# EXPERIMENTAL DESIGN PARAMETERS ===============================================

# Define experimental parameters
config <- list(
  # Cross-validation strategy
  n_folds = 5,              # Number of folds for k-fold CV
  n_cv_repeats = 10,        # Repeated CV iterations per experiment
  n_experiments = 5,        # Independent experiments with different seeds
  
  # Data splitting
  train_proportion = 0.75,  # Training set proportion
  test_proportion = 0.25,   # Test set proportion
  
  # Model hyperparameters
  rf_ntree = 500,           # Number of trees in RF
  rf_mtry = 10,             # Number of variables sampled at each split
  glmnet_alpha_seq = seq(0.05, 0.95, length.out = 10),  # GLMNet mixing
  
  # Grouped cross-validation (if applicable)
  group_variable = NULL,    # Set to column name if using grouped CV
  # e.g., "Experiment" for nested data
  
  # Random seeds for reproducibility
  random_seeds = c(100, 234, 567, 891, 1234)
)

# Validate configuration
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

# INITIALIZE RESULT STORAGE ====================================================

# Data frames for storing results
results_rmse <- data.frame()
results_classification <- data.frame()

# List for storing detailed CV metrics
cv_results_detailed <- list()

# MAIN EXPERIMENTAL LOOP =======================================================

cat("\n", rep("=", 80), "\n")
cat("STARTING EXPERIMENTAL PIPELINE\n")
cat(rep("=", 80), "\n\n")

for(experiment_id in 1:config$n_experiments) {
  
  cat("\n", rep("-", 80), "\n")
  cat("EXPERIMENT", experiment_id, "OF", config$n_experiments, "\n")
  cat("Random seed:", config$random_seeds[experiment_id], "\n")
  cat(rep("-", 80), "\n\n")
  
  # Set experiment-specific seed
  set.seed(config$random_seeds[experiment_id])
  
  # STEP 1: TRAIN-TEST SPLIT ---------------------------------------------------
  cat("Step 1: Creating train-test split...\n")
  
  if (!is.null(config$group_variable) && 
      config$group_variable %in% colnames(data)) {
    # Grouped splitting (ensures samples from same group stay together)
    cat("  Using grouped splitting by:", config$group_variable, "\n")
    train_indices <- groupdata2::partition(
      data, 
      p = config$train_proportion, 
      id_col = config$group_variable,
      list_out = FALSE
    )$partitions == 1
  } else {
    # Standard stratified splitting
    # FIXED: createDataPartition returns indices, need to convert to logical vector
    train_indices_numeric <- createDataPartition(
      y = data$Germination_Rate, 
      p = config$train_proportion, 
      list = FALSE
    )[, 1]
    
    # Create logical vector for proper subsetting
    train_indices <- rep(FALSE, nrow(data))
    train_indices[train_indices_numeric] <- TRUE
  }
  
  # Create training and testing datasets
  train_data <- data[train_indices, ]
  test_data <- data[!train_indices, ]
  
  cat("  Training samples:", nrow(train_data), "\n")
  cat("  Testing samples:", nrow(test_data), "\n")
  
  # Validate split
  if(nrow(test_data) == 0) {
    stop("ERROR: Test set is empty! Check data splitting logic.")
  }
  
  # Calculate classification threshold (median of training data)
  # Important: calculated only on training data to prevent data leakage
  classification_threshold <- median(train_data$Germination_Rate)
  cat("  Classification threshold (median):", 
      round(classification_threshold, 3), "\n\n")
  
  # STEP 2: FEATURE ENGINEERING ------------------------------------------------
  cat("Step 2: Feature engineering and preprocessing...\n")
  
  # Create preprocessing recipe
  preprocessing_recipe <- recipe(Germination_Rate ~ ., data = train_data) %>%
    # Remove Experiment ID if it exists
    step_rm(matches("^Experiment$")) %>%
    step_dummy(all_nominal_predictors()) %>%  # One-hot encoding
    prep()
  
  # Apply preprocessing
  train_x <- bake(preprocessing_recipe, new_data = NULL) %>% 
    select(-Germination_Rate) %>% 
    as.matrix()
  
  test_x <- bake(preprocessing_recipe, new_data = test_data) %>%
    select(-Germination_Rate) %>%
    as.matrix()
  
  train_y <- train_data$Germination_Rate
  test_y <- test_data$Germination_Rate
  
  cat("  Feature matrix dimensions:", dim(train_x)[1], "x", dim(train_x)[2], "\n\n")
  
  # STEP 3: GLMNet MODEL WITH NESTED CV ----------------------------------
  cat("Step 3: Training GLMNet model with nested cross-validation...\n")
  
  glmnet_cv_rmse <- numeric()
  
  # Nested cross-validation loop
  for(cv_iteration in 1:config$n_cv_repeats) {
    cat("  CV iteration", cv_iteration, "of", config$n_cv_repeats, "...")
    
    # Set iteration-specific seed
    set.seed(config$random_seeds[experiment_id] * 100 + cv_iteration)
    
    # Create CV folds
    if (!is.null(config$group_variable) && 
        config$group_variable %in% colnames(train_data)) {
      # Grouped folds
      cv_folds <- groupdata2::fold(
        train_data, 
        k = config$n_folds, 
        id_col = config$group_variable
      )$.folds
    } else {
      # Standard folds
      cv_folds <- createFolds(train_y, k = config$n_folds)
    }
    
    # Convert fold list to fold ID vector if necessary
    if(is.list(cv_folds)) {
      fold_id_vector <- rep(NA, nrow(train_data))
      for(fold_idx in seq_along(cv_folds)) {
        fold_id_vector[cv_folds[[fold_idx]]] <- fold_idx
      }
      cv_folds <- fold_id_vector
    }
    
    # Train GLMNet with alpha grid search
    cv_glmnet_model <- cva.glmnet(
      x = train_x,
      y = train_y,
      alpha = config$glmnet_alpha_seq,
      nfolds = config$n_folds,
      foldid = cv_folds,
      standardize = TRUE,
      type.measure = "mse"
    )
    
    # Extract best alpha and corresponding CV error
    best_alpha_idx <- which.min(sapply(cv_glmnet_model$modlist, 
                                       function(m) min(m$cvm)))
    best_cv_mse <- min(cv_glmnet_model$modlist[[best_alpha_idx]]$cvm)
    glmnet_cv_rmse <- c(glmnet_cv_rmse, sqrt(best_cv_mse))
    
    cat(" RMSE:", round(sqrt(best_cv_mse), 4), "\n")
  }
  
  # Train final GLMNet model on all training data
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
  
  # Predict on test set
  test_pred_glmnet <- predict(
    best_glmnet_model, 
    newx = test_x, 
    s = "lambda.1se"
  )[, 1]
  
  glmnet_test_rmse <- sqrt(mean((test_pred_glmnet - test_y)^2))
  
  cat("  GLMNet CV RMSE: mean =", round(mean(glmnet_cv_rmse), 4),
      "± SD =", round(sd(glmnet_cv_rmse), 4), "\n")
  cat("  GLMNet Test RMSE:", round(glmnet_test_rmse, 4), "\n\n")
  
  # STEP 4: RF MODEL WITH NESTED CV ---------------------------------
  cat("Step 4: Training RF model with nested cross-validation...\n")
  
  rf_cv_rmse <- numeric()
  
  # Nested cross-validation loop
  for(cv_iteration in 1:config$n_cv_repeats) {
    cat("  CV iteration", cv_iteration, "of", config$n_cv_repeats, "...")
    
    set.seed(config$random_seeds[experiment_id] * 100 + cv_iteration)
    
    # Create CV folds
    if (!is.null(config$group_variable) && 
        config$group_variable %in% colnames(train_data)) {
      cv_folds <- groupdata2::fold(
        train_data, 
        k = config$n_folds, 
        id_col = config$group_variable
      )$.folds
    } else {
      cv_folds <- createFolds(train_y, k = config$n_folds)
    }
    
    # Evaluate RF on each fold
    fold_rmse <- numeric(config$n_folds)
    for(fold_idx in 1:config$n_folds) {
      # Extract validation indices
      if(is.list(cv_folds)) {
        val_indices <- cv_folds[[fold_idx]]
      } else {
        val_indices <- which(cv_folds == fold_idx)
      }
      
      fold_train <- train_data[-val_indices, ]
      fold_val <- train_data[val_indices, ]
      
      # Train RF on fold
      rf_fold_model <- randomForest(
        Germination_Rate ~ . - Experiment,  # Exclude Experiment ID
        data = fold_train,
        ntree = config$rf_ntree,
        mtry = min(config$rf_mtry, ncol(fold_train) - 2),  # Adjust mtry if needed
        importance = FALSE
      )
      
      # Validate
      fold_predictions <- predict(rf_fold_model, newdata = fold_val)
      fold_rmse[fold_idx] <- sqrt(mean((fold_predictions - fold_val$Germination_Rate)^2))
    }
    
    rf_cv_rmse <- c(rf_cv_rmse, mean(fold_rmse))
    cat(" RMSE:", round(mean(fold_rmse), 4), "\n")
  }
  
  # Train final RF model on all training data
  cat("  Training final RF model...\n")
  final_rf_model <- randomForest(
    Germination_Rate ~ . - Experiment,  # Exclude Experiment ID
    data = train_data,
    ntree = config$rf_ntree,
    mtry = min(config$rf_mtry, ncol(train_data) - 2),  # Adjust mtry if needed
    importance = TRUE,
    keep.forest = TRUE
  )
  
  # Predict on test set
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
  
  # Store detailed CV results
  cv_results_detailed[[experiment_id]] <- list(
    glmnet_cv = glmnet_cv_rmse,
    rf_cv = rf_cv_rmse
  )
  
  # STEP 7: CLASSIFICATION METRICS ---------------------------------------------
  cat("Step 7: Computing binary classification metrics...\n")
  
  # Convert continuous predictions to binary classes
  test_class_true <- factor(
    ifelse(test_y >= classification_threshold, "High", "Low"),
    levels = c("High", "Low")
  )
  
  # Function to calculate confusion matrix metrics
  calculate_classification_metrics <- function(predictions, true_labels) {
    pred_classes <- factor(
      ifelse(predictions >= classification_threshold, "High", "Low"),
      levels = c("High", "Low")
    )
    
    confusion_matrix <- table(Predicted = pred_classes, Actual = true_labels)
    
    # Calculate proportions
    total <- sum(confusion_matrix)
    
    # Handle cases where confusion matrix might not have all categories
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
  
  # Calculate metrics for all models
  No_Model_metrics <- calculate_classification_metrics(
    No_Model_prediction, test_class_true
  )
  glmnet_metrics <- calculate_classification_metrics(
    test_pred_glmnet, test_class_true
  )
  rf_metrics <- calculate_classification_metrics(
    test_pred_rf, test_class_true
  )
  
  # Store classification results
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
  
  # STEP 8: SAVE FINAL MODELS FOR FEATURE IMPORTANCE --------------------------
  if(experiment_id == config$n_experiments) {
    final_glmnet_for_importance <- best_glmnet_model
    final_rf_for_importance <- final_rf_model
    final_train_x_matrix <- train_x
    final_train_data <- train_data
  }
  
  cat("\nExperiment", experiment_id, "completed.\n")
}

# RESULTS AGGREGATION AND STATISTICAL ANALYSIS ================================

cat("\n", rep("=", 80), "\n")
cat("RESULTS SUMMARY ACROSS ALL EXPERIMENTS\n")
cat(rep("=", 80), "\n\n")

# Compute summary statistics for RMSE
cat("--- Test Set RMSE Summary (", config$n_experiments, " experiments) ---\n", sep="")
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

# Calculate improvement over No_Model
cat("\n--- Model Improvement Over No_Model ---\n")
cat("GLMNet: ", 
    round((1 - rmse_summary$GLMNet_Mean / rmse_summary$No_Model_Mean) * 100, 2),
    "% reduction in RMSE\n", sep="")
cat("RF: ", 
    round((1 - rmse_summary$RF_Mean / rmse_summary$No_Model_Mean) * 100, 2),
    "% reduction in RMSE\n", sep="")

# Export detailed results
cat("\n--- Exporting Results ---\n")
write_csv(results_rmse, "Results_RMSE_Detailed.csv")
write_csv(results_classification, "Results_Classification_Detailed.csv")
cat("Results saved to CSV files.\n")

# PUBLICATION-QUALITY VISUALIZATIONS ===========================================

cat("\n", rep("=", 80), "\n")
cat("GENERATING PUBLICATION-QUALITY FIGURES\n")
cat(rep("=", 80), "\n\n")

# Define custom theme for Nature Communications style
theme_publication <- function(base_size = 14, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Text
      text = element_text(color = "black"),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(size = base_size - 2, color = "black"),
      
      # Axes
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.line = element_line(color = "black", linewidth = 0.5),
      
      # Panel
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      
      # Legend
      legend.position = "right",
      legend.title = element_text(face = "bold", size = base_size - 2),
      legend.text = element_text(size = base_size - 2),
      legend.key.size = unit(1, "lines"),
      
      # Plot
      plot.title = element_text(hjust = 0.5, face = "bold", size = base_size + 2),
      plot.subtitle = element_text(hjust = 0.5, size = base_size),
      plot.margin = unit(c(10, 10, 10, 10), "pt")
    )
}

# Color palette (Nature Communications style)
color_palette <- c(
  No_Model = "#999999",
  GLMNet = "#E69F00",
  RF = "#56B4E9",
  CV = "#009E73",
  Test = "#CC79A7"
)

# FIGURE 1: Cross-Validation vs Test Set Performance --------------------------
cat("Creating Figure 1: CV vs Test Set RMSE comparison...\n")

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
    Model = factor(
      Model, 
      levels = c("GLMNet", "RF"),
      labels = c("GLMNet", "RF")
    )
  )

figure1 <- ggplot(fig1_data, aes(x = Model, y = RMSE, fill = Type)) +
  geom_boxplot(
    position = position_dodge(0.8), 
    width = 0.6, 
    alpha = 0.8,
    outlier.shape = NA,
    linewidth = 1.6
  ) +
  geom_point(
    aes(color = Type),  # 添加颜色映射
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
    size = 3.5,
    alpha = 0.7,
    shape = 21,
    show.legend = FALSE  # 不显示散点的图例
  ) +
  scale_fill_manual(
    values = c(CV = "#6DADF9", Test = "#F6B293"),  # 使用指定颜色
    labels = c("Cross-Validation", "Test Set")
  ) +
  scale_color_manual(
    values = c(CV = "#6DADF9", Test = "#F6B293")  # 散点边框颜色与填充色一致
  ) +
  labs(
    title = "",
    x = "Model",
    y = "Root Mean Squared Error (RMSE)",
    fill = "Evaluation"
  ) +
  theme_publication() +
  theme(
    text = element_text(family = "Times New Roman", size = 30),
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.background = element_blank()  
  )

ggsave(
  "Figure1_CV_vs_Test_Performance.jpg", 
  figure1, 
  width = 10, 
  height = 6, 
  dpi = 600,
  units = "in"
)

ggsave(
  "Figure1_CV_vs_Test_Performance.pdf", 
  figure1, 
  width = 10, 
  height = 6,
  units = "in"
)


cat("  Saved: Figure1_CV_vs_Test_Performance.jpg/pdf\n")

# FIGURE 2: Model Comparison (Test Set Only) ----------------------------------
cat("Creating Figure 2: Test set RMSE comparison...\n")

fig2_data <- results_rmse %>%
  select(Experiment, No_Model_RMSE, GLMNet_Test_RMSE, RF_Test_RMSE) %>%
  pivot_longer(
    cols = -Experiment, 
    names_to = "Model", 
    values_to = "RMSE"
  ) %>%
  mutate(
    Model = factor(
      Model,
      levels = c("No_Model_RMSE", "GLMNet_Test_RMSE", "RF_Test_RMSE"),
      labels = c("No_Model", "GLMNet", "RF")
    )
  )

figure2 <- ggplot(fig2_data, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot(width = 0.6, alpha = 0.8, outlier.shape = NA) +
  geom_jitter(
    width = 0.15, 
    size = 3, 
    alpha = 0.6,
    shape = 21,
    color = "black"
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 4,
    fill = "red",
    color = "black"
  ) +
  scale_fill_manual(values = color_palette) +
  labs(
    title = "Test Set Performance Comparison",
    subtitle = paste0(config$n_experiments, " independent experiments with different random seeds"),
    x = "",
    y = "Root Mean Squared Error (RMSE)"
  ) +
  theme_publication() +
  theme(
    text = element_text(family = "Times New Roman", size = 30),
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.background = element_blank(), 
    legend.position = "none"
  )

ggsave(
  "Figure2_Test_Set_Comparison.jpg", 
  figure2, 
  width = 8, 
  height = 6, 
  dpi = 600,
  units = "in"
)


ggsave(
  "Figure2_Test_Set_Comparison.pdf", 
  figure2, 
  width = 8, 
  height = 6, 
  dpi = 600,
  units = "in"
)


cat("  Saved: Figure2_Test_Set_Comparison.jpg/pdf\n")

# FIGURE 3: Classification Performance ----------------------------------------
cat("Creating Figure 3: Classification metrics...\n")

fig3_data <- results_classification %>%
  mutate(
    Model = factor(
      Model,
      levels = c("No_Model", "GLMNet", "RF"),
      labels = c("No_Model", "GLMNet", "RF")
    )
  ) %>%
  group_by(Model) %>%
  summarise(
    across(c(True_Positive, False_Positive, True_Negative, False_Negative), mean)
  ) %>%
  pivot_longer(
    cols = -Model, 
    names_to = "Metric", 
    values_to = "Proportion"
  ) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c("True_Positive", "False_Positive", "True_Negative", "False_Negative"),
      labels = c("True Positive", "False Positive", "True Negative", "False Negative")
    )
  )

figure3 <- ggplot(fig3_data, aes(x = Model, y = Proportion, fill = Metric)) +
  geom_bar(stat = "identity", position = "stack", width = 0.7) +
  geom_text(
    aes(label = sprintf("%.2f", Proportion)),
    position = position_stack(vjust = 0.5),
    size = 3.5,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "True Positive" = "#01607A",
      "False Positive" = "#6DADF9",
      "True Negative" = "#F6B293",
      "False Negative" = "#B72234"
    )
  ) +
  labs(
    title = "Binary Classification Performance",
    subtitle = "Average confusion matrix proportions across experiments",
    x = "",
    y = "Proportion",
    fill = "Classification"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_publication()+
  theme(
    text = element_text(family = "Times New Roman", size = 30),
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.background = element_blank()  
  )

ggsave(
  "Figure3_Classification_Metrics.jpg", 
  figure3, 
  width = 10, 
  height = 6, 
  dpi = 600,
  units = "in"
)
ggsave(
  "Figure3_Classification_Metrics.pdf", 
  figure3, 
  width = 10, 
  height = 6,
  units = "in"
)

cat("  Saved: Figure3_Classification_Metrics.jpg/pdf\n")

# FEATURE IMPORTANCE ANALYSIS ==================================================

cat("\n", rep("=", 80), "\n")
cat("FEATURE IMPORTANCE ANALYSIS\n")
cat(rep("=", 80), "\n\n")

# Identify bacterial strain features (columns 2-21)
strain_feature_names <- colnames(data)[2:21]
cat("Analyzing importance of", length(strain_feature_names), "bacterial strains...\n")

# RF FEATURE IMPORTANCE --------------------------------------------
cat("\nExtracting RF feature importance...\n")

rf_importance_raw <- importance(final_rf_for_importance)[, "%IncMSE", drop = FALSE]
rf_importance_normalized <- (rf_importance_raw / sum(abs(rf_importance_raw))) * 100

rf_importance_df <- data.frame(
  Feature = rownames(rf_importance_raw),
  RF_Importance = as.numeric(rf_importance_normalized)
) %>%
  arrange(desc(RF_Importance))

cat("  Top 5 features (RF):\n")
print(head(rf_importance_df, 5))

# GLMNet FEATURE IMPORTANCE ----------------------------------------------
cat("\nExtracting GLMNet feature importance...\n")

glmnet_coefficients <- as.vector(
  coef(final_glmnet_for_importance, s = "lambda.1se")
)[-1]  # Remove intercept

glmnet_importance_raw <- data.frame(
  Feature = colnames(final_train_x_matrix),
  Coefficient = abs(glmnet_coefficients)
)

# Aggregate dummy variable importance back to original features
glmnet_importance_aggregated <- sapply(strain_feature_names, function(feature) {
  sum(glmnet_importance_raw$Coefficient[
    grep(paste0("^", feature, "(_|$)"), glmnet_importance_raw$Feature)
  ])
})

glmnet_importance_normalized <- (glmnet_importance_aggregated / 
                                   sum(glmnet_importance_aggregated)) * 100

glmnet_importance_df <- data.frame(
  Feature = names(glmnet_importance_normalized),
  GLMNet_Importance = as.numeric(glmnet_importance_normalized)
) %>%
  arrange(desc(GLMNet_Importance))

cat("  Top 5 features (GLMNet):\n")
print(head(glmnet_importance_df, 5))

# COMBINE IMPORTANCE SCORES ---------------------------------------------------
cat("\nCombining feature importance scores...\n")

importance_combined <- merge(
  rf_importance_df, 
  glmnet_importance_df, 
  by = "Feature", 
  all = TRUE
)
importance_combined[is.na(importance_combined)] <- 0

# Filter for strain features only
importance_strains <- importance_combined %>%
  filter(Feature %in% strain_feature_names) %>%
  arrange(desc((RF_Importance + GLMNet_Importance) / 2))

cat("  Combined importance calculated for", nrow(importance_strains), "strains.\n")

# Export importance scores
write_csv(importance_strains, "Feature_Importance_Scores.csv")
cat("  Saved: Feature_Importance_Scores.csv\n")

# FIGURE 4: Feature Importance Heatmap ----------------------------------------
cat("\nCreating Figure 4: Feature importance heatmap...\n")

fig4_data <- importance_strains %>%
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
    Feature = factor(Feature, levels = rev(importance_strains$Feature))
  )

figure4 <- ggplot(fig4_data, aes(x = Model, y = Feature, fill = Importance)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(
    aes(label = sprintf("%.1f%%", Importance)),
    size = 3,
    fontface = "bold"
  ) +
  scale_fill_gradientn(
    colors = c("#FFFFFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C", "#08306B"),
    values = scales::rescale(c(0, 1, 5, 10, 20, 40)),
    name = "Importance\n(%)",
    guide = guide_colorbar(
      barwidth = 1.5,
      barheight = 15,
      title.position = "top"
    )
  ) +
  labs(
    title = "Feature Importance Analysis",
    subtitle = "Relative importance of bacterial strains in predicting germination rate",
    x = "",
    y = "Bacterial Strain"
  ) +
  theme_publication() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    panel.grid = element_blank()
  )

ggsave(
  "Figure4_Feature_Importance.jpg", 
  figure4, 
  width = 8, 
  height = 12, 
  dpi = 600,
  units = "in"
)
ggsave(
  "Figure4_Feature_Importance.svg", 
  figure4, 
  width = 8, 
  height = 12,
  units = "in"
)

cat("  Saved: Figure4_Feature_Importance.jpg/pdf\n")

# SUPPLEMENTARY FIGURE: CV Stability ------------------------------------------
cat("\nCreating Supplementary Figure: CV stability across iterations...\n")

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

supp_figure <- ggplot(cv_stability_data, aes(x = Iteration, y = CV_RMSE, color = Model)) +
  geom_line(alpha = 0.6, linewidth = 0.8) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~ Experiment, ncol = 5, labeller = label_both) +
  scale_color_manual(
    values = c(
      "GLMNet" = color_palette["GLMNet"],
      "RF" = color_palette["RF"]
    )
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
    text = element_text(family = "Times New Roman", size = 30),
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.background = element_blank(),  
    strip.text = element_text(face = "bold")
  )


ggsave(
  "Supplementary_Figure_CV_Stability.jpg", 
  supp_figure, 
  width = 14, 
  height = 6, 
  dpi = 600,
  units = "in"
)
ggsave(
  "Supplementary_Figure_CV_Stability.pdf", 
  supp_figure, 
  width = 14, 
  height = 6,
  units = "in"
)

cat("  Saved: Supplementary_Figure_CV_Stability.jpg/pdf\n")

# STATISTICAL SIGNIFICANCE TESTING ============================================

cat("\n", rep("=", 80), "\n")
cat("STATISTICAL SIGNIFICANCE TESTING\n")
cat(rep("=", 80), "\n\n")

# Paired t-tests (since same test sets are used)
cat("Conducting paired t-tests on test set RMSE...\n\n")

# GLMNet vs No_Model
t_test_glmnet_No_Model <- t.test(
  results_rmse$GLMNet_Test_RMSE,
  results_rmse$No_Model_RMSE,
  paired = TRUE
)

cat("GLMNet vs No_Model:\n")
cat("  Mean difference:", round(t_test_glmnet_No_Model$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_glmnet_No_Model$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_glmnet_No_Model$p.value, digits = 3), "\n\n")

# RF vs No_Model
t_test_rf_No_Model <- t.test(
  results_rmse$RF_Test_RMSE,
  results_rmse$No_Model_RMSE,
  paired = TRUE
)

cat("RF vs No_Model:\n")
cat("  Mean difference:", round(t_test_rf_No_Model$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_rf_No_Model$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_rf_No_Model$p.value, digits = 3), "\n\n")

# RF vs GLMNet
t_test_rf_glmnet <- t.test(
  results_rmse$RF_Test_RMSE,
  results_rmse$GLMNet_Test_RMSE,
  paired = TRUE
)

cat("RF vs GLMNet:\n")
cat("  Mean difference:", round(t_test_rf_glmnet$estimate, 4), "\n")
cat("  t-statistic:", round(t_test_rf_glmnet$statistic, 3), "\n")
cat("  p-value:", format.pval(t_test_rf_glmnet$p.value, digits = 3), "\n\n")

# Export statistical test results
statistical_tests <- data.frame(
  Comparison = c(
    "GLMNet vs No_Model",
    "RF vs No_Model",
    "RF vs GLMNet"
  ),
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
  ),
  CI_Lower = c(
    t_test_glmnet_No_Model$conf.int[1],
    t_test_rf_No_Model$conf.int[1],
    t_test_rf_glmnet$conf.int[1]
  ),
  CI_Upper = c(
    t_test_glmnet_No_Model$conf.int[2],
    t_test_rf_No_Model$conf.int[2],
    t_test_rf_glmnet$conf.int[2]
  )
)

write_csv(statistical_tests, "Statistical_Test_Results.csv")
cat("Statistical test results saved to: Statistical_Test_Results.csv\n")

# FINAL SUMMARY AND SESSION INFO ==============================================

cat("\n", rep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("Summary of outputs:\n")
cat("  - CSV files:\n")
cat("    * Results_RMSE_Detailed.csv\n")
cat("    * Results_Classification_Detailed.csv\n")
cat("    * Feature_Importance_Scores.csv\n")
cat("    * Statistical_Test_Results.csv\n")
cat("  - Figures (JPG and PDF):\n")
cat("    * Figure1_CV_vs_Test_Performance\n")
cat("    * Figure2_Test_Set_Comparison\n")
cat("    * Figure3_Classification_Metrics\n")
cat("    * Figure4_Feature_Importance\n")
cat("    * Supplementary_Figure_CV_Stability\n\n")

cat("Session information:\n")
print(sessionInfo())

cat("\n", rep("=", 80), "\n")
cat("All analyses completed successfully!\n")
cat("Results are ready for manuscript submission.\n")
cat(rep("=", 80), "\n\n")

################################################################################
# END OF SCRIPT
################################################################################
