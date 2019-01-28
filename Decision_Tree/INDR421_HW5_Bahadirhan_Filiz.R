# Author: Bahadirhan Filiz
# Implementation of decision tree regression algorithm

# read data into memory
data_set <- read.csv("hw05_data_set.csv")
train_data <- data_set[1:100,]
test_data <- data_set[101:133,]
x_train <- train_data$x
y_train <- train_data$y
x_test <- test_data$x
y_test <- test_data$y

# get numbers of train and test samples
N_train <- length(x_train)
N_test <- length(x_test)

# P parameter
P <- 10

# create necessary data structures
node_indices <- list()
is_terminal <- c()
need_split <- c()
node_splits <- c()



# put all training instances into the root node
node_indices <- list(1:N_train)
is_terminal <- c(FALSE)
need_split <- c(TRUE)


# learning algorithm
while (1) {
  # find nodes that need splitting
  split_nodes <- which(need_split)
  # check whether we reach all terminal nodes
  if (length(split_nodes) == 0) {
    break
  }
  
  # find best split positions for all nodes
  for (split_node in split_nodes) {
    
    data_indices <- node_indices[[split_node]]
    need_split[split_node] <- FALSE
    # check whether node's number of elements are under P
    if (length(data_indices) <= P) {
      is_terminal[split_node] <- TRUE
    } else {
      is_terminal[split_node] <- FALSE
      
      unique_values <- sort(unique(x_train[data_indices]))
      split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
      split_scores <- rep(0, length(split_positions))
      for (s in 1:length(split_positions)) {
        left_indices <- data_indices[which(x_train[data_indices] < split_positions[s])]
        right_indices <- data_indices[which(x_train[data_indices] >= split_positions[s])]
        g_left = mean(y_train[left_indices])
        g_right = mean(y_train[right_indices])
        split_scores[s] <-   sum((y_train[left_indices]-g_left)^2) +  sum((y_train[right_indices]-g_right)^2) 
      }
      best_score <- min(split_scores)
      best_split <- split_positions[which.min(split_scores)]
      node_splits[split_node] <- best_split

      
      # create left node using the selected split
      left_indices <- data_indices[which(x_train[data_indices] < best_split)]
      node_indices[[2 * split_node]] <- left_indices
      is_terminal[2 * split_node] <- FALSE
      need_split[2 * split_node] <- TRUE
      
      # create left node using the selected split
      right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
      node_indices[[2 * split_node + 1]] <- right_indices
      is_terminal[2 * split_node + 1] <- FALSE
      need_split[2 * split_node + 1] <- TRUE
    }
  }
}

# plot
plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylab = "y", xlab = "x")
points(x_test, y_test, type = "p", pch = 19, col = "red")


splits_pos <- node_splits[which(node_splits>0)]
splits_pos <- c(splits_pos, 0, 60)
splits_pos <- sort(splits_pos)


indexes <- c()
g_s <- c()
for (i in 1:(length(splits_pos)-1)) {
  y1 <- splits_pos[i]
  y2 <- splits_pos[i+1]
  indexes <- which(x_train <= y2 & x_train > y1)
  g_s[i] <- mean(y_train[indexes])
}

data_interval <- c()
for (i in 1:(length(splits_pos)-1)) {
  data_interval <- c(data_interval, seq(from = splits_pos[i], to = splits_pos[i+1], by = 0.01))
}

y_interval <- c()
for (i in 1:(length(splits_pos)-1)) {
  y_interval <- c(y_interval, rep(g_s[i], length(seq(from = splits_pos[i], to = splits_pos[i+1], by = 0.01))))
}

lines(data_interval, y_interval, lwd = 2, col = "black")

# RMSE
RMSE_test <- 0
for (i in 1:length(g_s)) {
  y1 <- splits_pos[i]
  y2 <- splits_pos[i+1]
  indexes <- which(x_test <= y2 & x_test > y1)
  RMSE_test <- RMSE_test + sum((y_test[indexes] - g_s[i])^2)
}
RMSE_test <- sqrt(RMSE_test/N_test)


















# Second Part P 1:20
RMSE_running <- rep(0,20)
for (P in 1:20) {
  # create necessary data structures
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()
  node_splits <- c()
  
  
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      # check whether node's number of elements are under P
      if (length(data_indices) <= P) {
        is_terminal[split_node] <- TRUE
      } else {
        is_terminal[split_node] <- FALSE
        
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] < split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] >= split_positions[s])]
          g_left = mean(y_train[left_indices])
          g_right = mean(y_train[right_indices])
          split_scores[s] <-   sum((y_train[left_indices]-g_left)^2) +  sum((y_train[right_indices]-g_right)^2) 
        }
        best_score <- min(split_scores)
        best_split <- split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        
        # create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create left node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  
  
  
  splits_pos <- node_splits[which(node_splits>0)]
  splits_pos <- c(splits_pos, 0, 60)
  splits_pos <- sort(splits_pos)
  
  
  indexes <- c()
  g_s <- c()
  for (i in 1:(length(splits_pos)-1)) {
    y1 <- splits_pos[i]
    y2 <- splits_pos[i+1]
    indexes <- which(x_train <= y2 & x_train > y1)
    g_s[i] <- mean(y_train[indexes])
  }
  
  # RMSE
  for (i in 1:length(g_s)) {
    y1 <- splits_pos[i]
    y2 <- splits_pos[i+1]
    indexes <- which(x_test <= y2 & x_test > y1)
    RMSE_running[P] <- RMSE_running[P] + sum((y_test[indexes] - g_s[i])^2)
  }
  RMSE_running[P] <- sqrt(RMSE_running[P]/N_test)
  
  
  
}
P <- 1:20
plot(P, RMSE_running, type="b", lwd = 2)


sprintf("RMSE is %f when P is 10", RMSE_test)





