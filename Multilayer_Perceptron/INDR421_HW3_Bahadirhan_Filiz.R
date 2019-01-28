# Author: Bahadirhan Filiz
# Implementation multilayer perceptron for multiclass discrimination
# Has single layer of hidden nodes 
# Sigmoid Activation function on Output Layer
# Sigmoid Activation function on Hidden Layer



# importing data sets:
images_set = read.csv(file = "hw03_data_set_images.csv", header = FALSE)
y_truth_all = read.csv(file = "hw03_data_set_labels.csv", header = FALSE)
training_index = c(1:25, 40:64, 79:103, 118:142, 157:181)

# X ve y_truth
X = images_set[training_index, ]
X = as.matrix(X)
y_truth = y_truth_all[training_index, ]

# number of classes and obs
K = 5
N = nrow(X)
D = 320

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1


# safelog:
safelog <- function(x) {
  return (log(x + 1e-100))
}

# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# define the softmax function
softmax <- function(Z, V) {
  scores <- cbind(1, Z) %*% V
  scores <- exp(scores - matrix(apply(scores, MARGIN = 2, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}


# set learning parameters
eta <- 0.005
epsilon <- 1e-3
H <- 20
max_iteration <- 200


# randomly initalize W and v
set.seed(521)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
V <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H + 1, K)

Z = sigmoid(cbind(1,X) %*% W)
Y_predicted = softmax(Z, V)

objective_values <- -sum(Y_truth * safelog(Y_predicted)) 
iteration <- 1
while(1) {

  
  V = V + eta * (t(cbind(1,Z)) %*% (Y_truth - Y_predicted))
  
  W = W + eta * (t(cbind(1,X)) %*% (((Y_truth - Y_predicted) %*% t(V[-1, ])) * Z * (1-Z)))
  
  Z = sigmoid(cbind(1,X) %*% W)
  Y_predicted = softmax(Z, V)
  objective_values <-  c(objective_values, -sum(Y_truth * safelog(Y_predicted)) )
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(V)


# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


# calculate confusion matrix for training data
y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)



# calculate confusion matrix for test data
test_index = c(26:39, 65:78, 104:117, 143:156, 182:195)
X_test = images_set[test_index, ]
X_test = as.matrix(X_test)
y_truth_test = y_truth_all[test_index, ]
Z_test = sigmoid(cbind(1,X_test) %*% W)
Y_predicted_test <- softmax(Z_test, V)
y_predicted_test <- apply(Y_predicted_test, 1, which.max)
confusion_matrix_test <- table(y_predicted_test, y_truth_test)
print(confusion_matrix_test)








