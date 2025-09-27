# Machine Learning
# Final Project
# May 18th, 2025

library(ggplot2)
library(dplyr)
library(tidyr)
library(splines)
library(Rmisc)
library(MASS)

setwd("/Applications/IMC463/finalproject")

tiktok = read.csv("tiktok_sample_25.csv")
str(tiktok)

summary(tiktok)

# Factorize vars 
tiktok$gender.deepface = as.factor(tiktok$gender.deepface)
tiktok$race = as.factor(tiktok$race)
tiktok$Verified.Status= as.factor(tiktok$Verified.Status)

# Add log-transformed variables using log1p() safely
tiktok$log_Share.Count   		        <- log1p(tiktok$Share.Count )
tiktok$log_mean_comment_disapproval <- log1p(tiktok$mean_comment_disapproval)
tiktok$log_mean_comment_approval    <- log1p(tiktok$mean_comment_approval)
tiktok$log_mean_comment_admiration  <- log1p(tiktok$mean_comment_admiration)
tiktok$log_mean_comment_disgust     <- log1p(tiktok$mean_comment_disgust)
tiktok$log_mean_comment_anger       <- log1p(tiktok$mean_comment_anger)
tiktok$log_Follower.Count           <- log1p(tiktok$Follower.Count)
tiktok$log_Likes.sum.Count        	<- log1p(tiktok$Likes.sum.Count)
tiktok$log_Following.Count        	<- log1p(tiktok$Following.Count)

# Check distribution 
hist(tiktok$image_quality)

# Negative Binomial Model (raw)
nb_model <- glm.nb(Share.Count ~ mean_comment_approval + mean_comment_disapproval + 
                     mean_comment_anger + mean_comment_admiration + mean_comment_disgust,
                   data = tiktok)
summary(nb_model)

# Negative Binomial Model (both IVs & DVs log-transformed)
tiktok <- tiktok %>%
  mutate(log_Share_Count = log(Share.Count + 1))  # log(Share.Count + 1) to avoid log(0) issues


log_nb <- glm.nb(log_Share_Count ~ log(mean_comment_approval + 1) + log(mean_comment_disapproval + 1) + 
         log(mean_comment_anger + 1) + log(mean_comment_admiration + 1) + 
         log(mean_comment_disgust + 1), data = tiktok, link = "log")

summary(log_nb)



# Negative Binomial Model (IVs log-transformed)
log_nb2 <- glm.nb(Share.Count ~ log(mean_comment_approval + 1) + log(mean_comment_disapproval + 1) + 
                   log(mean_comment_anger + 1) + log(mean_comment_admiration + 1) + 
                   log(mean_comment_disgust + 1), data = tiktok, link = "log")

summary(log_nb2)

# Regression Tree
library(rpart)
library(rpart.plot)

set.seed(123)

# Fit regression tree
tree <- rpart(log_Share.Count ~ log_mean_comment_disapproval+ log_mean_comment_approval + log_mean_comment_admiration
             + log_mean_comment_anger + age + log_Follower.Count + log_Likes.sum.Count + log_Following.Count + 
               Brightness + Sharpness, data = tiktok,
              control = rpart.control(cp = 0.01,
                                      minsplit = 20, 
                                      minbucket = 7)) 
summary(tree)

rpart.plot(tree, 
           type = 1, 
           extra = 101,
           box.palette = "Purples",
           main = "Regression Tree: Predicting TikTok Video Share Count")

print(tree$variable.importance)

tree_sentiment <- rpart(log_Share.Count ~ log_mean_comment_disapproval + log_mean_comment_approval +
                          log_mean_comment_admiration + log_mean_comment_disgust + log_mean_comment_anger,
                        data = tiktok,
                        control = rpart.control(cp = 0.01, minsplit = 20, minbucket = 7))

rpart.plot(tree_sentiment, 
           type = 1, 
           extra = 101,
           box.palette = "Blues",
           main = "Regression Tree: Predicting TikTok Video Share Count")

print(tree_sentiment$variable.importance)

#Random Forest

library(randomForest)
library(caret)

set.seed(12345)  # For reproducibility

# Split data into training and testing (optional, since OOB error is used)
train_idx <- sample(1:nrow(tiktok), 0.7 * nrow(tiktok))
train_data <- tiktok[train_idx, ]
test_data <- tiktok[-train_idx, ]

# Train Random Forest (using all data, since OOB error is built-in)
rf_model <- randomForest(
  log_Share.Count ~ log_mean_comment_disapproval+ log_mean_comment_approval + log_mean_comment_admiration
  + log_mean_comment_anger + age + log_Follower.Count + log_Likes.sum.Count + log_Following.Count + 
    Brightness + Sharpness,
  data = train_data,
  ntree = 500,
  mtry = 5,  # or try tuneGrid later
  importance = TRUE,
  na.action = na.omit
)

# Print model summary
print(rf_model)

# Extract OOB error
oob_error <- rf_model$err.rate

# Plot OOB error vs. number of trees
plot(rf_model, main = "OOB Error vs. Number of Trees")

# Get OOB predictions
oob_pred <- rf_model$predicted

# OOB predictions (regression)
oob_pred <- rf_model$predicted
actual <- rf_model$y

# Regression evaluation
#mse <- mean((oob_pred - actual)^2)
#rmse <- sqrt(mse)
#mae <- mean(abs(oob_pred - actual))
#r2 <- cor(oob_pred, actual)^2

#cat("OOB RMSE:", round(rmse, 2), "\n")
#cat("OOB MAE:", round(mae, 2), "\n")
#cat("OOB R-squared:", round(r2, 4), "\n")

# Variable importance
varImpPlot(rf_model, main = "Variable Importance")

importance_values <- importance(rf_model)
print(importance_values)

# Define the relevant variables
vars_rf <- c("log_Share.Count", 
             "log_mean_comment_disapproval", "log_mean_comment_approval", 
             "log_mean_comment_admiration", "log_mean_comment_anger", "age", 
             "log_Follower.Count", "log_Likes.sum.Count", "log_Following.Count", 
             "Brightness", "Sharpness")

# Remove rows with NAs in those columns
train_data_clean <- train_data[complete.cases(train_data[, vars_rf]), ]
test_data_clean <- test_data[complete.cases(test_data[, vars_rf]), ]

# Now use caret for tuning
library(caret)

set.seed(123)
tune_grid <- expand.grid(.mtry = c(15, 18, 23, 25))  # Customize mtry values

rf_tuned <- train(
  log_Share.Count ~ log_mean_comment_disapproval+ log_mean_comment_approval + 
    log_mean_comment_admiration + log_mean_comment_anger + age + 
    log_Follower.Count + log_Likes.sum.Count + log_Following.Count + 
    Brightness + Sharpness,
  data = train_data_clean,
  method = "rf",
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 5),
  ntree = 200
)


print(rf_tuned$bestTune) # Print the best mtry value
plot(rf_tuned) # Plot cross-validation results

# Retrain with best hyperparameters
final_rf <- randomForest(
  log_Share.Count ~ log_mean_comment_disapproval+ log_mean_comment_approval + log_mean_comment_admiration
  + log_mean_comment_anger + age + log_Follower.Count + log_Likes.sum.Count + log_Following.Count + 
    Brightness + Sharpness,
  data = train_data_clean,
  ntree = 200,
  mtry = rf_tuned$bestTune$mtry,
  importance = TRUE
)

print(final_rf) # Print final model results

plot(final_rf)

# Get predictions on the test set
predictions <- predict(final_rf, newdata = test_data_clean)

# Calculate MSE (mean squared error) on the test set
mse_test <- mean((predictions - test_data_clean$log_Share.Count)^2)
print(paste("Test MSE:", mse_test))

## Lasso 
library(glmnet)

# Include log_Share.Count in the cleaning process
lasso_data <- tiktok[, c("log_Share.Count", 
                         "log_mean_comment_disapproval", "log_mean_comment_approval", 
                         "log_mean_comment_admiration", "log_mean_comment_disgust",
                         "log_mean_comment_anger", "age", "gender.deepface", "race", 
                         "log_Follower.Count", "log_Likes.sum.Count", "log_Following.Count", 
                         "Verified.Status", "log_image_quality", "log_Brightness", 
                         "log_Sharpness")]

# Drop any rows with missing values and convert to data frame
lasso_data_clean <- na.omit(lasso_data)
lasso_data_clean <- as.data.frame(lasso_data_clean)

# Prepare predictors and outcome
preds_lasso <- as.matrix(lasso_data_clean[, -1])  # exclude outcome column
outcome_lasso <- lasso_data_clean$log_Share.Count

set.seed(123)  # For reproducibility

# Step 1: Run 10-fold CV LASSO
cv_lasso <- cv.glmnet(preds_lasso, outcome_lasso, alpha = 1)  # Use correct vars
plot(cv_lasso)

# Step 2: Get best lambda values
best_lambda_lasso <- cv_lasso$lambda.min
simpler_lambda_lasso <- cv_lasso$lambda.1se

# Step 3: Fit final LASSO model using best lambda
final_lasso <- glmnet(preds_lasso, outcome_lasso, alpha = 1, lambda = best_lambda_lasso)

# Step 4: Extract and view coefficients
coefficients_lasso <- coef(final_lasso)
print(coefficients_lasso)


library(dplyr)
library(Rmisc)
library(gridExtra)


# Create bins of log_mean_comment_disapproval (e.g., quartiles)
tiktok <- tiktok %>%
  mutate(
    bin_disapproval = ntile(log_mean_comment_disapproval, 4),
    bin_approval = ntile(log_mean_comment_approval, 4),
    bin_anger = ntile(log_mean_comment_anger, 4),
    bin_admiration = ntile(log_mean_comment_admiration, 4),
    bin_disgust = ntile(log_mean_comment_disgust, 4),
    bin_follower = ntile(log_Follower.Count, 4),
    bin_like = ntile(log_Likes.sum.Count, 4)
  )

# Summarize log_Share.Count by disapproval_bin
summ_disapproval <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_disapproval")
summ_approval <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_approval")
summ_anger <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_anger")
summ_admiration <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_admiration")
summ_disgust<- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_disgust")

summ_follower <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_follower")
summ_like <- summarySE(tiktok, measurevar = "log_Share.Count", groupvars = "bin_like")

# 3. Create plots
plot_bar <- function(df, xvar, title) {
  ggplot(df, aes_string(x = xvar, y = "log_Share.Count")) +
    geom_bar(stat = "identity", fill = "#56B4E9") +
    geom_errorbar(aes(ymin = log_Share.Count - se, ymax = log_Share.Count + se), width = 0.2) +
    labs(x = title, y = "Mean log(Share Count)") +
    theme_minimal()
}

p1 <- plot_bar(summ_disapproval, "as.factor(bin_disapproval)", "Mean Disapproval")
p2 <- plot_bar(summ_approval, "as.factor(bin_approval)", "Mean Approval")
p3 <- plot_bar(summ_anger, "as.factor(bin_anger)", "Mean Anger")
p4 <- plot_bar(summ_admiration, "as.factor(bin_admiration)", "Mean Admiration")
p5 <- plot_bar(summ_disgust, "as.factor(bin_disgust)", "Mean Disgust")
p6 <- plot_bar(summ_follower, "as.factor(bin_follower)", "Follower Count")
p7 <- plot_bar(summ_like, "as.factor(bin_like)", "Like Count")

# 4. Arrange in 3x2 grid
grid.arrange(p1, p2, p3, p4, p5, ncol = 3)


