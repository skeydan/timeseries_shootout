source("common.R")

model_exists <- FALSE

# since we are using stateful rnn tsteps can be set to 1
lstm_num_timesteps <- 1
batch_size <- 5
epochs <- 500
lstm_units <- 4
lstm_type <- "stateful"
data_type <- "data_raw"
test_type <- "TREND"

(model_name <- paste(test_type, "_model_lstm_simple", lstm_type, data_type, epochs, "epochs", sep="_"))

# get data into "timesteps form"
X_train <- t(sapply(1:(length(trend_train) - lstm_num_timesteps), function(x) trend_train[x:(x + lstm_num_timesteps - 1)]))
dim(X_train)

y_train <- sapply((lstm_num_timesteps + 1):(length(trend_train)), function(x) trend_train[x])
length(y_train)

X_test <- t(sapply(1:(length(trend_test) - lstm_num_timesteps), function(x) trend_test[x:(x + lstm_num_timesteps - 1)]))
y_test <- sapply((lstm_num_timesteps + 1):(length(trend_test)), function(x) trend_test[x])


# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
dim(X_train) <- c(dim(X_train)[1], dim(X_train)[2], 1)
dim(X_train)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]
c(num_samples, num_steps, num_features)

dim(X_test) <- c(dim(X_test)[1], dim(X_test)[2], 1)

model %>% 
  layer_lstm(units = lstm_units, batch_input_shape = c(batch_size, num_steps, num_features), stateful = TRUE)
  
# model
if (!model_exists) {
  set.seed(22222)
  model <- keras_model_sequential() 
  model %>% 
    layer_lstm(units = lstm_units, input_shape = c(num_steps, num_features), stateful = TRUE,
               batch_size=batch_size) %>% 
    layer_dense(units = 1) %>% 
    compile(
      loss = 'mean_squared_error',
      optimizer = 'adam'
    )
  
  model %>% summary()
  
  for (i in 1:epochs) {
    model %>% fit(X_train, y_train, batch_size = batch_size,
                  epochs = 1, verbose = 1, shuffle = FALSE)
    
    model %>% reset_states()
  }
  
  model %>% save_model_hdf5(filepath = paste0(model_name, ".h5"))
} else {
  model <- load_model_hdf5(filepath = paste0(model_name, ".h5"))
}

pred_train <- model %>% predict(X_train, batch_size = batch_size)
pred_test <- model %>% predict(X_test, batch_size = batch_size)

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps), pred_train, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps), pred_test))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

# test on in-range test set
trend_test <- trend_test_inrange
X_test <- t(sapply(1:(length(trend_test) - lstm_num_timesteps), function(x) trend_test[x:(x + lstm_num_timesteps - 1)]))
dim(X_test) <- c(dim(X_test)[1], dim(X_test)[2], 1)
y_test <- sapply((lstm_num_timesteps + 1):(length(trend_test)), function(x) trend_test[x])
pred_test <- model %>% predict(X_test, batch_size = batch_size)

(test_rsme <- sqrt(sum((tail(trend_test,length(trend_test) - lstm_num_timesteps) - pred_test)^2)))

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps), pred_train, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps), pred_test))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

