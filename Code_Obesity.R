# ------------------------------
# Load Necessary Libraries
# ------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)       # Added tidyr for pivot_longer
library(reshape2)
library(viridis)
library(gridExtra)
library(ggpubr)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(xgboost)
library(e1071)
library(nnet)
library(doParallel)
library(RColorBrewer)
library(psych)
library(cluster)     # For silhouette methods
library(knitr)       # For pretty tables
library(kableExtra)

# ------------------------------
# Enable Parallel Processing
# ------------------------------
cl <- makeCluster(detectCores() - 1)  # Use all but one core
registerDoParallel(cl)

# ------------------------------
# Load the Dataset
# ------------------------------
file_path <- "C:\\Users\\Sara\\Desktop\\ObesityDataSet_raw_and_data_sinthetic.csv"
dataset <- read.csv(file_path, sep = ",", header = TRUE, stringsAsFactors = FALSE)
head(dataset)

# ------------------------------
# Data Cleaning and Preprocessing
# ------------------------------
missing_values <- sum(is.na(dataset))
cat("Total missing values:", missing_values, "\n")

if (missing_values > 0) {
  dataset <- na.omit(dataset)
}

duplicate_rows <- sum(duplicated(dataset))
cat("Total duplicate rows:", duplicate_rows, "\n")

if (duplicate_rows > 0) {
  dataset <- dataset[!duplicated(dataset), ]
}

colnames(dataset) <- c(
  "Gender", "Age", "Height", "Weight", 
  "FamHist", "FAVC", "VegCons", "Meals", 
  "FoodFreq", "Smoke", "WaterIntake", "CaloricSurplus", 
  "PhysActFreq", "TechUsage", "AlcoholCons", "Transport", 
  "ObesityLevel"
)

# Convert categorical variables to numeric
dataset$Gender <- ifelse(tolower(dataset$Gender) == "male", 1, 0)
binary_vars <- c("FamHist", "FAVC", "Smoke", "CaloricSurplus")
dataset[binary_vars] <- lapply(dataset[binary_vars], function(x) {
  ifelse(tolower(x) == "yes" | tolower(x) == "frequently", 1, 0)
})

dataset$AlcoholCons <- case_when(
  tolower(dataset$AlcoholCons) == "no" ~ 0,
  tolower(dataset$AlcoholCons) == "sometimes" ~ 1,
  tolower(dataset$AlcoholCons) == "frequently" ~ 2,
  TRUE ~ NA_real_
)

if (any(is.na(dataset$AlcoholCons))) {
  dataset$AlcoholCons[is.na(dataset$AlcoholCons)] <- median(dataset$AlcoholCons, na.rm = TRUE)
}

dataset$FoodFreq <- case_when(
  tolower(dataset$FoodFreq) == "never" ~ 0,
  tolower(dataset$FoodFreq) == "rarely" ~ 1,
  tolower(dataset$FoodFreq) == "sometimes" ~ 2,
  tolower(dataset$FoodFreq) == "frequently" ~ 3,
  tolower(dataset$FoodFreq) == "always" ~ 4,
  TRUE ~ NA_real_
)

if (any(is.na(dataset$FoodFreq))) {
  dataset$FoodFreq[is.na(dataset$FoodFreq)] <- median(dataset$FoodFreq, na.rm = TRUE)
}

dataset$Transport <- ifelse(tolower(dataset$Transport) %in% c("public_transportation", "automobile"), 0, 
                            ifelse(tolower(dataset$Transport) %in% c("bike", "walking"), 1, NA))

if (any(is.na(dataset$Transport))) {
  dataset$Transport[is.na(dataset$Transport)] <- 0
}

dataset$ObesityLevel <- case_when(
  tolower(dataset$ObesityLevel) %in% c("overweight_level_i", "overweight_level_ii") ~ 1,
  tolower(dataset$ObesityLevel) %in% c("underweight", "normal_weight") ~ 0,
  TRUE ~ NA_real_
)

if (any(is.na(dataset$ObesityLevel))) {
  dataset$ObesityLevel[is.na(dataset$ObesityLevel)] <- median(dataset$ObesityLevel, na.rm = TRUE)
}

# Scale numeric variables using Min-Max Scaling
numeric_vars <- sapply(dataset, is.numeric)

min_max_scale <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

dataset_scaled <- dataset
dataset_scaled[, numeric_vars] <- lapply(dataset_scaled[, numeric_vars], min_max_scale)
dataset_clustering <- dataset_scaled
dataset_scaled$ObesityLevel <- factor(dataset_scaled$ObesityLevel, levels = c(0, 1))

head(dataset_scaled)
str(dataset_scaled)

str(dataset_clustering)

# ------------------------------
# Descriptive Statistics
# ------------------------------

# 1. Summary Statistics for Numeric Variables
numeric_summary <- dataset_scaled %>%
  select(where(is.numeric)) %>%
  summarise_all(list(
    Mean = ~mean(.),
    Median = ~median(.),
    SD = ~sd(.),
    Min = ~min(.),
    Max = ~max(.)
  )) %>%
  pivot_longer(cols = everything(),
               names_to = c("Variable", ".value"),
               names_sep = "_")

# Display Summary Statistics Table with Soft Pink Header
cat("\n**Summary Statistics for Numeric Variables:**\n")
kable(numeric_summary, caption = "Summary Statistics") %>%
  kable_styling(full_width = FALSE, position = "left") %>%
  row_spec(0, background =  "#d4edda",, bold = TRUE)  # Soft pink header


# 2. Frequency Tables for Categorical Variables
categorical_vars <- dataset_scaled %>%
  select(ObesityLevel) %>%
  names()

frequency_tables <- lapply(categorical_vars, function(var) {
  dataset_scaled %>%
    group_by(!!sym(var)) %>%
    summarise(Count = n()) %>%
    mutate(Percentage = round((Count / sum(Count)) * 100, 2))
})

names(frequency_tables) <- categorical_vars

# Display Frequency Tables with Soft Pink Header
for (var in categorical_vars) {
  cat(paste0("\n**Frequency Table for ", var, ":**\n"))
  print(
    kable(frequency_tables[[var]], caption = paste("Frequency of", var)) %>%
      kable_styling(full_width = FALSE, position = "left") %>%
      row_spec(0, background = "#d4edda",, bold = TRUE)  # Soft pink header
  )
}


# Heatmap
# 1. Compute the Correlation Matrix for Numeric Variables
# Select only numeric columns from dataset_scaled
numeric_data <- dataset_scaled[, sapply(dataset_scaled, is.numeric)]

# Calculate the correlation matrix using pairwise complete observations
cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")

# 2. Melt the Correlation Matrix into Long Format
cor_matrix_long <- melt(cor_matrix, varnames = c("Variable1", "Variable2"), value.name = "Correlation")

# 3. Create the Correlation Heatmap using ggplot2
heatmap_plot <- ggplot(data = cor_matrix_long, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile(color = "white") +  # Add white borders between tiles
  scale_fill_viridis_c(direction = 1, option = "turbo") +  # Use viridis color scale with 'cividis' option for blue/green tones
  geom_text(aes(label = round(Correlation, 2)), color = "white", size = 3) +  # Annotate tiles with correlation values
  labs(title = "Correlation Heatmap of Numeric Variables",
       x = "Variables",
       y = "Variables",
       fill = "Correlation") +  # Add labels and title
  theme_minimal() +  # Use a minimal theme
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Rotate x-axis labels for readability
    axis.text.y = element_text(size = 10),  # Adjust y-axis label size
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14)  # Style the plot title
  )

# 4. Display the Heatmap
print(heatmap_plot)



# ------------------------------
# Supervised Learning Analysis
# ------------------------------

# Split the dataset into training and testing sets
set.seed(111)
indexes <- createDataPartition(dataset_scaled$ObesityLevel, p = 0.80, list = FALSE)
train_set <- dataset_scaled[indexes, ]
test_set <- dataset_scaled[-indexes, ]

################### Logistic Regression ###################

# Build the logistic regression model
glm_model <- glm(ObesityLevel ~ ., family = binomial, data = train_set)
summary(glm_model)

# Predict on test set
predicted_probabilities <- predict(glm_model, newdata = test_set, type = "response")
threshold <- 0.5
predicted_classes <- ifelse(predicted_probabilities > threshold, 1, 0)

# Actual classes
actual_classes <- as.numeric(as.character(test_set$ObesityLevel))

# Confusion Matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
accuracy_log_reg <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("The accuracy for Logistic Regression model is:", round(accuracy_log_reg, 4), "\n")

# Precision and Recall
predicted_classes_factor <- factor(predicted_classes, levels = c(0, 1))
actual_classes_factor <- factor(actual_classes, levels = c(0, 1))

conf_matrix <- confusionMatrix(predicted_classes_factor, actual_classes_factor, positive = "1")
print(conf_matrix)
precision_log_reg <- conf_matrix$byClass['Pos Pred Value']
recall_log_reg <- conf_matrix$byClass['Sensitivity']
F1_log_reg <- 2 * (precision_log_reg * recall_log_reg) / (precision_log_reg + recall_log_reg)

cat("Precision for Logistic Regression:", round(precision_log_reg, 4), "\n")
cat("Recall for Logistic Regression:", round(recall_log_reg, 4), "\n")
cat("F1 Score for Logistic Regression:", round(F1_log_reg, 4), "\n")

################### K-Nearest Neighbors (KNN) ###################

train_control_knn <- trainControl(method = "cv", number = 10)
set.seed(111)
knn_model <- train(ObesityLevel ~ ., data = train_set, method = "knn",
                   trControl = train_control_knn, tuneLength = 10)
knn_predictions <- predict(knn_model, newdata = test_set)

# Confusion Matrix
cm_knn <- confusionMatrix(knn_predictions, test_set$ObesityLevel, positive = "1")
cat("Confusion Matrix for KNN:\n")
print(cm_knn)

# Calculate Evaluation Metrics
accuracy_knn <- cm_knn$overall["Accuracy"]
precision_knn <- cm_knn$byClass["Pos Pred Value"]
recall_knn <- cm_knn$byClass["Sensitivity"]
F1_knn <- 2 * (precision_knn * recall_knn) / (precision_knn + recall_knn)

cat("Accuracy:", round(accuracy_knn, 4), "\n")
cat("Precision:", round(precision_knn, 4), "\n")
cat("Recall:", round(recall_knn, 4), "\n")
cat("F1 Score:", round(F1_knn, 4), "\n")

################### Decision Tree ###################

# Train control for cross-validation
train_control_tree <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# Define the tuning grid for complexity parameter (cp)
tune_grid_tree <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

# Train the Decision Tree model
tree_model <- train(ObesityLevel ~ ., data = train_set, method = "rpart",
                    trControl = train_control_tree, tuneGrid = tune_grid_tree)

# Adjust the decision tree plot for smaller squares
# Display only split variable and predicted class
prp(tree_model$finalModel,
    type = 1,               # Show only the split variable in nodes
    extra = 0,              # Do not display class probabilities or percentages
    fallen.leaves = TRUE,   # Place leaves at the bottom of the tree
    cex = 0.6,              # Decrease font size for compactness
    tweak = 1,              # Adjust width and spacing of nodes
    box.palette = "auto",   # Use default color palette
    varlen = 0,             # Do not truncate variable names
    faclen = 0)             # Do not truncate factor levels

# Predicting the test set
tree_predictions <- predict(tree_model, newdata = test_set)

# Confusion Matrix
cm_tree <- confusionMatrix(tree_predictions, test_set$ObesityLevel, positive = "1")

# Display the confusion matrix
cat("Confusion Matrix for Decision Tree:\n")
print(cm_tree)

# Calculate evaluation metrics
accuracy_tree <- cm_tree$overall["Accuracy"]
precision_tree <- cm_tree$byClass["Pos Pred Value"]
recall_tree <- cm_tree$byClass["Sensitivity"]
F1_tree <- 2 * (precision_tree * recall_tree) / (precision_tree + recall_tree)

# Display metrics
cat("Accuracy:", round(accuracy_tree, 4), "\n")
cat("Precision:", round(precision_tree, 4), "\n")
cat("Recall:", round(recall_tree, 4), "\n")
cat("F1 Score:", round(F1_tree, 4), "\n")

################### Random Forest ###################

train_control_rf <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
tune_grid_rf <- expand.grid(mtry = c(2, 4, 6, 8))

set.seed(123)
rf_model <- train(ObesityLevel ~ ., data = train_set, method = "rf",
                  trControl = train_control_rf, tuneGrid = tune_grid_rf, ntree = 500)

rf_predictions <- predict(rf_model, newdata = test_set)
cm_rf <- confusionMatrix(rf_predictions, test_set$ObesityLevel, positive = "1")

# Calculate Evaluation Metrics
accuracy_rf <- cm_rf$overall["Accuracy"]
precision_rf <- cm_rf$byClass["Pos Pred Value"]
recall_rf <- cm_rf$byClass["Sensitivity"]
F1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

cat("Accuracy:", round(accuracy_rf, 4), "\n")
cat("Precision:", round(precision_rf, 4), "\n")
cat("Recall:", round(recall_rf, 4), "\n")
cat("F1 Score:", round(F1_rf, 4), "\n")

# Variable Importance Analysis

# Extract Variable Importance using caret's varImp function
var_imp_rf <- varImp(rf_model, scale = FALSE)
print(var_imp_rf)

# Convert variable importance to a data frame for ggplot2
var_imp_df <- data.frame(
  Variable = rownames(var_imp_rf$importance),
  Importance = var_imp_rf$importance[,1]
)

# Arrange the variables in descending order of importance
var_imp_df <- var_imp_df %>%
  arrange(desc(Importance)) %>%
  top_n(20, Importance) %>%  # Adjust the number to display top N variables
  mutate(Variable = factor(Variable, levels = rev(unique(Variable))))

# Plot Variable Importance using ggplot2
ggplot(var_imp_df, aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "cadetblue") +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Variable Importance - Random Forest",
       x = "Variables",
       y = "Importance") +
  theme_minimal()

################### XGBoost with Parameter Tuning ###################

# Prepare trainControl
train_control_xgb <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Define tuning grid
tune_grid_xgb <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1),
  gamma = c(0, 0.1),
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(1, 5),
  subsample = c(0.7, 0.9)
)

# Train XGBoost model
set.seed(111)
xgb_tuned_model <- train(
  ObesityLevel ~ .,
  data = train_set,
  method = "xgbTree",
  trControl = train_control_xgb,
  tuneGrid = tune_grid_xgb,
  verbose = FALSE
)

# Display the best tuning parameters
cat("Best tuning parameters for XGBoost:\n")
print(xgb_tuned_model$bestTune)

# Make predictions on the test set
xgb_predictions <- predict(xgb_tuned_model, newdata = test_set)

# Confusion Matrix
cm_xgb <- confusionMatrix(xgb_predictions, test_set$ObesityLevel, positive = "1")

# Print the confusion matrix and performance metrics
cat("Confusion Matrix for XGBoost with Parameter Tuning:\n")
print(cm_xgb)

# Calculate Evaluation Metrics
accuracy_xgb <- cm_xgb$overall["Accuracy"]
precision_xgb <- cm_xgb$byClass["Pos Pred Value"]
recall_xgb <- cm_xgb$byClass["Sensitivity"]
F1_xgb <- 2 * (precision_xgb * recall_xgb) / (precision_xgb + recall_xgb)

cat("Accuracy:", round(accuracy_xgb, 4), "\n")
cat("Precision:", round(precision_xgb, 4), "\n")
cat("Recall:", round(recall_xgb, 4), "\n")
cat("F1 Score:", round(F1_xgb, 4), "\n")

# ------------------------------
# Comparison of Models
# ------------------------------

# Create a data frame with the metrics
metrics_df <- data.frame(
  Model = c("Logistic Regression", "KNN", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(accuracy_log_reg, accuracy_knn, accuracy_tree, accuracy_rf, accuracy_xgb),
  Precision = c(precision_log_reg, precision_knn, precision_tree, precision_rf, precision_xgb),
  Recall = c(recall_log_reg, recall_knn, recall_tree, recall_rf, recall_xgb),
  F1_Score = c(F1_log_reg, F1_knn, F1_tree, F1_rf, F1_xgb)
)

# Display the metrics table with soft green header
library(kableExtra)
library(knitr)
kable(metrics_df, digits = 4, caption = "Model Comparison") %>%
  kable_styling(full_width = FALSE, position = "center") %>%
  row_spec(0, background = "#d4edda", bold = TRUE)  # Soft green header



# ------------------------------
# Stop Parallel Processing
# ------------------------------
stopCluster(cl)
registerDoSEQ()
