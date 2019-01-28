# Author: Bahadirhan Filiz
# Implementations of three nonparametric regression algorithms
# 1) Regressogram
# 2) Running Mean Smoother
# 3) Gaussian Kernel

# read data into memory
data_set <- read.csv("hw04_data_set.csv")
train_data <- data_set[1:100,]
test_data <- data_set[101:133,]

minimum_value <- 0
maximum_value <- 60
data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.01)

# Regressogram 
bin_width <- 3
left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)
p_head <- sapply(1:length(left_borders), function(b) { mean(train_data[which(left_borders[b] < train_data$x & train_data$x <=right_borders[b]),2])} )
binNox <- function(x) {floor(x/bin_width) + 1}
p_headx <- function(b) { mean(train_data[which((binNox(b) * bin_width - bin_width) < train_data$x & train_data$x <= binNox(b) * bin_width ),2])
}

# plot
plot(train_data$x, train_data$y, type = "p", pch = 19, col = "blue",
     xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x")
points(test_data$x, test_data$y, type = "p", pch = 19, col = "red")
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

# RMSE cal
sum <- 0
for (c in 1:nrow(test_data)) {
  sum = sum + (p_headx(test_data$x[c]) - test_data$y[c])^2
}
RMSE_regressogram <- sqrt(sum/nrow(test_data))


# Running Mean Smoother
p_head <- sapply(data_interval, function(b) { mean(train_data[which((b - 0.5 * bin_width) < train_data$x & train_data$x <= (b + 0.5 * bin_width)),2])} )
p_headx <- function(b) { mean(train_data[which((b - 0.5 * bin_width) < train_data$x & train_data$x <= (b + 0.5 * bin_width)),2])}
# plot
plot(train_data$x, train_data$y, type = "p", pch = 19, col = "blue",
     xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x")
points(test_data$x, test_data$y, type = "p", pch = 19, col = "red")
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

# RMSE cal
sum <- 0
for (c in 1:nrow(test_data)) {
  sum = sum + (p_headx(test_data$x[c]) - test_data$y[c])^2
}
RMSE_running <- sqrt(sum/nrow(test_data))



# Kernel Smoother
bin_width <- 1
p_head <- sapply(data_interval, function(x) {sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - train_data$x)^2 / bin_width^2) * train_data$y) / sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - train_data$x)^2 / bin_width^2))})     
p_headx <- function(x) {sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - train_data$x)^2 / bin_width^2) * train_data$y) / sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - train_data$x)^2 / bin_width^2))}
  



# plot
plot(train_data$x, train_data$y, type = "p", pch = 19, col = "blue",
     xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x")
points(test_data$x, test_data$y, type = "p", pch = 19, col = "red")
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

# RMSE cal
sum <- 0
for (c in 1:nrow(test_data)) {
  sum = sum + (p_headx(test_data$x[c]) - test_data$y[c])^2
}
RMSE_kernel <- sqrt(sum/nrow(test_data))


cat("Regressogram => RMSE is", RMSE_regressogram ,"when h is 3")
cat("Running Mean Smoother => RMSE is", RMSE_running ,"when h is 3")
cat("Kernel Smoother => RMSE is" , RMSE_kernel ," when h is 1")









