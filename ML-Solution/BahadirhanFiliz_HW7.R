# Author: Bahadirhan Filiz
# Machine learning solution for a real-life binary classification problem from finance industry
# Predicsts whether a customer will answer the phone call initiated by the outbound call center of a bank 
# Using the information given about each particular customer and call time



# Design:
# Compares logistic regression and combination of (decision tree, logistic regression) 
# by using ROC Curve Approach on K-Folds Cross Validation



# libraries
library(tree)
library(AUC)
library(class)
library(ROCR)

# set seed
set.seed(421)

# Read data
X_train <- read.csv("training_data.csv", header = TRUE)
X_test <- read.csv("test_data.csv", header = TRUE)
y_train <- as.factor(read.csv("training_labels.csv", header = FALSE)[,1])

D <- ncol(X_train)
K <- 2

# insight information about distribution of classes
summary(y_train)
negative_count <- length(which(y_train == 0))
positive_count <- length(which(y_train == 1))
positive_ratio <- positive_count/(negative_count + positive_count)
negative_ratio <- negative_count/(negative_count + positive_count)



# K-Fold Cross Validation - each fold has equal amount of classes to perform better
Fold_number <- 5
positive_indexes <- which(y_train == 1)
negative_indexes <- which(y_train == 0)
Fold_list <- list()

for (i in 1:Fold_number) {
  print(sprintf("folding progress #%d", i))
  positive_fold_indexes <- sample(positive_indexes, (positive_count/Fold_number))
  negative_fold_indexes <- sample(negative_indexes, (negative_count/Fold_number))
  Fold_indexes <- sample(c(positive_fold_indexes, negative_fold_indexes))
  Fold_list[[i]] <- Fold_indexes
}

auc_values <- c()
auc_values_combined <- c()
for (i in 1:Fold_number) {
  print(sprintf("cross validating #%d", i))
  X_train_cont_currentFold <- X_train[Fold_list[[i]],]                     # takes current folds's training data
  y_train_cont_currentFold <- y_train[Fold_list[[i]]]
  X_test_currentFold <- X_train[-Fold_list[[i]],]                          # takes current fold's test data
  y_test_currentFold <- y_train[-Fold_list[[i]]]
  X_test_currentFold <- X_train[-Fold_list[[i]],]
  y_test_currentFold <- y_train[-Fold_list[[i]]]
  
  model <- glm(y ~., family=binomial(link='logit'), data = cbind(X_train_cont_currentFold, y = y_train_cont_currentFold))
  fitted.results <- predict(model, newdata= X_test_currentFold, type="response")
  guess_logit <- ifelse(fitted.results > 0.5,1,0)
  
  pr <- prediction(fitted.results, y_test_currentFold)
  prf <- performance(pr, measure = "tpr", x.measure = "fpr")
  plot(prf, main = sprintf("ROC CURVE FOR K-FOLD #%s", i))
  
  auc <- performance(pr, measure = "auc")
  auc <- auc@y.values[[1]]
  auc_values <- c(auc, auc_values)
  
  
  tree_classifier <- tree(y ~ ., data = cbind(X_train_cont_currentFold, y = y_train_cont_currentFold))
  ptree <- predict(tree_classifier, X_test_currentFold)
  ptree <- ptree[,2]
  
  combined <- (0.5*ptree+0.5*fitted.results)
  guess_combined <- ifelse(fitted.results > 0.5,1,0)
  roc_curve <- roc(combined, y_test_currentFold)
  auc_values_combined <- c(auc(roc_curve), auc_values_combined)
  plot(roc_curve$fpr, roc_curve$tpr, main = sprintf("COMBINED ROC CURVE FOR K-FOLD #%s", i), lwd = 2, col = "blue", type = "b", las = 1)
  
  mean(auc_values_combined)
  mean(auc_values)
  
}


model_final <- glm(y ~., family=binomial(link='logit'), data = cbind(X_train, y = y_train))
fitted.results_final <- predict(model_final, newdata= X_test, type="response")
guess_logit_final <- ifelse(fitted.results > 0.5,1,0)


write.table(fitted.results_final, file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)



































# # Lets seperate continus data and binary data from each other  (some kind of artificial feature selection)
# binary <- c()
# for (i in 1:D) {
#   binary[i] <- TRUE
#   for (j in 1:1000) {
#     if(X_train[j,i] > 1) 
#       binary[i] <- FALSE
#   }
# }
# 
# X_train_binary <- X_train[,binary]
# X_train_cont <- X_train[,!binary]
# 
# 
# for (i in 1:Fold_number) {
#   print(sprintf("training decision tree #%d", i))
#   X_train_cont_currentFold <- X_train[-Fold_list[[i]],]                   # takes current folds's training data
#   y_train_cont_currentFold <- y_train[-Fold_list[[i]]]
#   X_test_currentFold <- X_train[Fold_list[[i]],]                          # takes current fold's test data
#   y_test_currentFold <- y_train[Fold_list[[i]]]
# }
# 
# 
# N <- 1000
# # k-nn on the continious data
# A <- knn(X_train_binary[N,], X_test_currentFold[N,], y_train_cont_currentFold[N,], k = 3, prob = TRUE)


# decision tree on continious data



# for (i in 1:Fold_number) {
#   X_train_currentFold <- X_train[-Fold_list[[i]],]                   # takes current folds's training data
#   y_train <- y_train[-Fold_list[[i]]]
#   X_test_currentFold <- X_train[Fold_list[[i]],]                     # takes current fold's test data
#   y_test <- y_train[Fold_list[[i]]]
# }



