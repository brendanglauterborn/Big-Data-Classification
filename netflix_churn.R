
install.packages(c("tidyverse","psych","GGally"))
library(tidyverse)
library(psych)
library(GGally)
install.packages("randomForest")
library(randomForest)
install.packages("Metrics")
library(Metrics)
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
install.packages("gbm")
library(gbm)
install.packages("DataExplorer")
library(DataExplorer)
install.packages("SmartEDA")
library(SmartEDA)
library(ggplot2)

install.packages("e1071")
library(e1071)


#init
raw_netflix_data <- read.csv("netflix_customer_churn.csv")
netflix_data <- subset(raw_netflix_data, select = -customer_id)

netflix_data$subscription_type <- as.integer(factor(netflix_data$subscription_type,levels = c("Basic", "Standard", "Premium"),
                                                    ordered = TRUE)) - 1L

netflix_data$gender         <- as.factor(netflix_data$gender)
netflix_data$region         <- as.factor(netflix_data$region)
netflix_data$device         <- as.factor(netflix_data$device)
netflix_data$payment_method <- as.factor(netflix_data$payment_method)
netflix_data$favorite_genre <- as.factor(netflix_data$favorite_genre)

X_cat <- model.matrix(~ gender + region + device + payment_method + favorite_genre, data = netflix_data, na.action = na.pass)

X_cat <- X_cat[,colnames(X_cat) != "(Intercept)", drop = FALSE]

netflix_data <- cbind(netflix_data, as.data.frame(X_cat))

set.seed(123)
n <- nrow(netflix_data)
train_idx <- sample(seq_len(n), size = 0.7*n)
train <- netflix_data[train_idx, ]
test  <- netflix_data[-train_idx, ]
train$churned <- as.factor(train$churned)
test$churned  <- as.factor(test$churned)

glm_model <- glm(churned ~ . - gender - region - device - payment_method - favorite_genre,
                 data = train,
                 family = binomial)
summary(glm_model)
glm_prob <- predict(glm_model, newdata = test, type = "response")
glm_pred <- ifelse(glm_prob > .5, "1", "0")
glm_pred <- factor(glm_pred, levels = levels(test$churned))
install.packages("caret")
library(caret)
confusionMatrix(glm_pred, test$churned)

library(pROC)
cm <- confusionMatrix(glm_pred, test$churned, positive = '1')
accuracy <- cm$overall["Accuracy"]
precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]
f1 <- 2 / ((1 / precision) + (1 / recall))


# ---- Heatmap of the confusion matrix ----
library(ggplot2)

cm_raw <- table(Predicted = glm_pred, Actual = test$churned)
cm_df  <- as.data.frame(cm_raw)

ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(title = "Confusion Matrix Heatmap – Logistic Regression",
       x = "Actual Class", 
       y = "Predicted Class", 
       fill = "Count") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))

roc_obj <- roc(response = test$churned, predictor = glm_prob, levels = rev(levels(test$churned)))
auc_val <- auc(roc_obj)

plot(roc_obj,
     col = "steelblue",
     lwd = 2,
     main = paste("ROC Curve - Logistic Regression (AUC =", round(auc_val, 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")  # diagonal reference line

#SVM 
svm_model <- svm(churned ~ . - gender - region - device - payment_method - favorite_genre,
                 data = train,
                 kernel = "radial",
                 cost = 1,
                 probability = TRUE)
svm_pred <- predict(svm_model, newdata = test,probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[,"1"]

# --- Confusion matrix & metrics ---
cm_svm <- confusionMatrix(svm_pred, test$churned, positive = "1")


# Accuracy / F1 (optional)
accuracy_svm  <- cm_svm$overall["Accuracy"]
precision_svm <- cm_svm$byClass["Pos Pred Value"]
recall_svm    <- cm_svm$byClass["Sensitivity"]
f1_svm        <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)

# --- Heatmap of confusion matrix ---
cm_tbl <- table(Predicted = svm_pred, Actual = test$churned)  # build a 2x2 table
cm_df  <- as.data.frame(cm_tbl)

ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(title = "Confusion Matrix Heatmap – SVM",
       x = "Actual Class", y = "Predicted Class", fill = "Count") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))

# --- ROC / AUC (optional but recommended) ---
roc_svm <- roc(response = test$churned, predictor = svm_prob, levels = rev(levels(test$churned)))
auc_svm <- auc(roc_svm)
plot(roc_svm, col = "steelblue", lwd = 2,
     main = paste("SVM ROC Curve (AUC =", round(auc_svm, 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")


#random forest
# Columns we actually want for RF (raw factors + numerics)
rf_cols <- c("churned",
             "gender","region","device","payment_method","favorite_genre",
             "subscription_type","age","watch_hours","last_login_days",
             "monthly_fee","number_of_profiles","avg_watch_time_per_day")

train_rf <- train[, rf_cols]
test_rf  <- test[, rf_cols]

set.seed(123)
rf_model <- randomForest(churned ~ ., data = train_rf,
                         ntree = 500,
                         mtry = sqrt(ncol(train_rf)-1),
                         importance = TRUE)

rf_pred <- predict(rf_model, newdata = test_rf)
rf_prob <- predict(rf_model, newdata = test_rf, type = "prob")[,"1"]

cm_rf <- confusionMatrix(rf_pred, test_rf$churned, positive = "1")
cm_rf

# --- Accuracy / Precision / Recall / F1 ---
accuracy_rf  <- cm_rf$overall["Accuracy"]
precision_rf <- cm_rf$byClass["Pos Pred Value"]
recall_rf    <- cm_rf$byClass["Sensitivity"]
f1_rf        <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

# --- Heatmap of confusion matrix ---
cm_tbl_rf <- table(Predicted = rf_pred, Actual = test_rf$churned)  # 2x2 table
cm_df_rf  <- as.data.frame(cm_tbl_rf)

ggplot(cm_df_rf, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(title = "Confusion Matrix Heatmap – Random Forest",
       x = "Actual Class", 
       y = "Predicted Class", 
       fill = "Count") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))

# --- ROC / AUC ---
roc_rf <- roc(response = test_rf$churned,
              predictor = rf_prob,
              levels = rev(levels(test_rf$churned)))
auc_rf <- auc(roc_rf)

plot(roc_rf, col = "steelblue", lwd = 2,
     main = paste("Random Forest ROC Curve (AUC =", round(auc_rf, 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")


auc_rf <- auc(roc(response = test_rf$churned,
                  predictor = rf_prob,
                  levels = rev(levels(test_rf$churned))))
auc_rf

varImpPlot(rf_model, main = "Random Forest Feature Importance")

