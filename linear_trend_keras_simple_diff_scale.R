source("common.R")

model_exists <- TRUE

lstm_num_timesteps <- 4 # one less now
batch_size <- 1
epochs <- 500
lstm_units <- 4
lstm_type <- "stateless"
data_type <- "data_diffed_scaled"
test_type <- "TREND"

(model_name <- paste(test_type, "_model_lstm_simple", lstm_type, data_type, epochs, "epochs", sep="_"))

# difference
trend_train_start <- trend_train[1]
trend_test_start <- trend_test[1]

trend_train_diff <- diff(trend_train)
trend_test_diff <- diff(trend_test)

# normalize
minval <- min(trend_train_diff)
maxval <- max(trend_train_diff)

normalize <- function(vec, min, max) {
  (vec-min) / (max-min)
}
denormalize <- function(vec,min,max) {
  vec * (max - min) + min
}

trend_train_diff <- normalize(trend_train_diff, minval, maxval)
trend_test_diff <- normalize(trend_test_diff, minval, maxval)


# get data into "timesteps form"
X_train <- t(sapply(1:(length(trend_train_diff) - lstm_num_timesteps), function(x) trend_train_diff[x:(x + lstm_num_timesteps - 1)]))
y_train <- sapply((lstm_num_timesteps + 1):(length(trend_train_diff)), function(x) trend_train_diff[x])
X_test <- t(sapply(1:(length(trend_test_diff) - lstm_num_timesteps), function(x) trend_test_diff[x:(x + lstm_num_timesteps - 1)]))
y_test <- sapply((lstm_num_timesteps + 1):(length(trend_test_diff)), function(x) trend_test_diff[x])

# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
dim(X_train) <- c(dim(X_train)[1], dim(X_train)[2], 1)
dim(X_train)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]
c(num_samples, num_steps, num_features)

dim(X_test) <- c(dim(X_test)[1], dim(X_test)[2], 1)

# model
if (!model_exists) {
  set.seed(22222)
  model <- keras_model_sequential() 
  model %>% 
    layer_lstm(units = lstm_units, input_shape = c(num_steps, num_features)) %>% 
    layer_dense(units = 1) %>% 
    compile(
      loss = 'mean_squared_error',
      optimizer = 'adam'
    )
  
  model %>% summary()
  
  model %>% fit( 
    X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = list(X_test, y_test)
  )
  model %>% save_model_hdf5(filepath = paste0(model_name, ".h5"))
} else {
  model <- load_model_hdf5(filepath = paste0(model_name, ".h5"))
}

pred_train <- model %>% predict(X_train, batch_size = 1)
pred_test <- model %>% predict(X_test, batch_size = 1)

pred_train <- denormalize(pred_train, minval, maxval)
pred_test <- denormalize(pred_test, minval, maxval)

pred_train_undiff <- pred_train + trend_train[(lstm_num_timesteps+1):(length(trend_train)-1)]
pred_test_undiff <- pred_test + trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

(test_rsme <- sqrt(sum((tail(trend_test,length(trend_test) - lstm_num_timesteps - 1) - pred_test_undiff)^2)))

# test on in-range dataset
trend_test <- trend_test_inrange
pred_test <- model %>% predict(X_test, batch_size = 1)
pred_test <- denormalize(pred_test, minval, maxval)

pred_train_undiff <- pred_train + trend_train[(lstm_num_timesteps+1):(length(trend_train)-1)]
pred_test_undiff <- pred_test + trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))
