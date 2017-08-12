#
build_model_name = function(model_type, test_type, lstm_type, data_type, epochs) {
  paste(model_type, lstm_type, data_type, test_type, epochs, "epochs", sep="_")
}

#
normalize <- function(vec, min, max) {
  (vec-min) / (max-min)
}
denormalize <- function(vec,min,max) {
  vec * (max - min) + min
}

# get data into "timesteps form": single matrix, for later chop-up into X and Y parts
build_matrix <- function(tseries, overall_timesteps) {
  X <- t(sapply(1:(length(tseries) - overall_timesteps + 1), #!!!!!!!!!! +1
             function(x) tseries[x:(x + overall_timesteps - 1)]))
  cat("\nBuilt matrix with dimensions: ", dim(X))
  return(X)
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
  cat("\nBuilt X matrix with dimensions: ", dim(X))
  return(X)
}

# get data into "timesteps form": target
build_y <- function(tseries, lstm_num_timesteps) {
  y <- sapply((lstm_num_timesteps + 1):(length(tseries)), function(x) tseries[x])
  cat("\nBuilt y vector with length: ", length(y))
  return(y)
}

# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  cat("\nReshaped X to dimensions: ", dim(X))
  return(X)
}
