# Author: Bahadirhan Filiz
# Implementation of naïve Bayes’ classifier for multiclass classification
# For black and white images of letters

# importing data sets:
images_set = read.csv(file = "hw01_data_set_images.csv", header = FALSE)
y_truth    = read.csv(file = "hw01_data_set_labels.csv", header = FALSE)

# subsetting training data:
A_training = images_set[1:25,]
B_training = images_set[40:64,]
C_training = images_set[79:103,]
D_training = images_set[118:142,]
E_training = images_set[157:181,]

# pij vectors:
pAj = colSums(A_training)/25
pBj = colSums(B_training)/25
pCj = colSums(C_training)/25
pDj = colSums(D_training)/25
pEj = colSums(E_training)/25

# safelog function
safelog <- function(x) {
  return(log(x+exp(-100)))
}

# Discriminant functions for general class using naive Bayesian classifier:
Discriminant_func <- function(data_vector, pij) {
  ones = rep(1, 320)
  score = as.matrix(data_vector) %*% as.matrix(safelog(pij)) + as.matrix(ones - data_vector) %*% as.matrix(safelog(ones - pij))
  return(drop(score))
}

# predictions y hat initialization:
y_hat = rep("X", 195)

# score functions in use & making predictions:
for (i in 1:195) {
  A_score = Discriminant_func(images_set[i,], pAj)
  B_score = Discriminant_func(images_set[i,], pBj)
  C_score = Discriminant_func(images_set[i,], pCj)
  D_score = Discriminant_func(images_set[i,], pDj)
  E_score = Discriminant_func(images_set[i,], pEj)
  winner = max(A_score, B_score, C_score, D_score, E_score)
  
  if ( A_score == winner ) {
    y_hat[i] = "A"
  }
  else if ( B_score == winner ) {
    y_hat[i] = "B"
  }
  else if ( C_score == winner ) {
    y_hat[i] = "C"
  }
  else if ( D_score == winner ) {
    y_hat[i] = "D"
  }
  else if ( E_score == winner ) {
    y_hat[i] = "E"
  }
}

# Confusion Matrix predictions compared with training data:
training_index = c(1:25, 40:64, 79:103, 118:142, 157:181)
y_truth_training = y_truth[training_index, ]
y_hat_training = y_hat[training_index]
CF_training = table(y_hat_training, y_truth_training)

# Confusion Matrix predictions compared with test data:
test_index = c(26:39, 65:78, 104:117, 143:156, 182:195)
y_truth_test = y_truth[test_index, ]
y_hat_test = y_hat[test_index]
CF_test = table(y_hat_test, y_truth_test)

# Display Confusion Matrices
CF_training
CF_test





