#
build_model_name = function(model_type, test_type, lstm_type, data_type, epochs) {
  paste(model_type, test_type, lstm_type, data_type, epochs, "epochs", sep="_")
}

# get data into "timesteps form": design matrix
build_X <- function(tseries, lstm_num_timesteps) {
  X <- if (lstm_num_timesteps > 1) {
    t(sapply(1:(length(tseries) - lstm_num_timesteps),
             function(x) tseries[x:(x + lstm_num_timesteps - 1)]))
  } else {
    tseries[1:length(tseries) - lstm_num_timesteps]
  }
  if (lstm_num_timesteps == 1) dim(X) <- c(length(X),1)
  message(paste0("Built X matrix with dimensions: ", dim(X)))
  return(X)
}

# get data into "timesteps form": target
build_y <- function(tseries, lstm_num_timesteps) {
  y <- sapply((lstm_num_timesteps + 1):(length(tseries)), function(x) tseries[x])
  message(paste0("Built y vector matrix with length: ", length(y)))
  return(y)
}

# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  message()
}
dim(X_train) <- 
dim(X_train)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]
c(num_samples, num_steps, num_features)

dim(X_test) <- c(dim(X_test)[1], dim(X_test)[2], 1)
