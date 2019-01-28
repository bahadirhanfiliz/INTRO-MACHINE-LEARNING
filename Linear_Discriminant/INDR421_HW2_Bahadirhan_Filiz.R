# Author: Bahadirhan Filiz
# Implementation of Linear Discriminant by Regression algorithm for multiclass classification
# For black and white images of letters

# importing data sets:
images_set = read.csv(file = "hw02_data_set_images.csv", header = FALSE)
y_truth_all = read.csv(file = "hw02_data_set_labels.csv", header = FALSE)
training_index = c(1:25, 40:64, 79:103, 118:142, 157:181)

# X ve y_truth
X = images_set[training_index, ]
X = as.matrix(X)
y_truth = y_truth_all[training_index, ]

# number of classes and obs
K = 5
N = nrow(X)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1


# safelog:
safelog <- function(x) {
  return (log(x + 1e-100))
}

# define the sigmoid function
sigmoid <- function(X, W, w0) {
  return (1 / (1 + exp(-(X %*% W + w0))))
}

# define the gradient functions
gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1 - Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted)))
}


# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize W and w0
set.seed(521)
W <- matrix(runif(ncol(X) * K, min = -0.01, max = 0.01), ncol(X), K)
w0 <- runif(K, min = -0.01, max = 0.01)


# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  Y_predicted <- sigmoid(X, W, w0)
  
  objective_values <- c(objective_values, 0.5 * sum((Y_truth -Y_predicted)^2))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(w0)


# plot objective function during iterations
plot(1:iteration, objective_values,
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
Y_predicted_test <- sigmoid(X_test, W, w0)
y_predicted_test <- apply(Y_predicted_test, 1, which.max)
confusion_matrix_test <- table(y_predicted_test, y_truth_test)
print(confusion_matrix_test)
