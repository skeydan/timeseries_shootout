source("common.R")
source("functions.R")

model_exists <- TRUE

lstm_num_timesteps <- 5
batch_size <- 1
epochs <- 500
lstm_units <- 4
model_type <- "model_lstm_simple"
lstm_type <- "stateless"
data_type <- "data_raw"
test_type <- "TREND"

model_name <- build_model_name(model_type, test_type, lstm_type, data_type, epochs)

cat("\n####################################################################################")
cat("\nRunning model: ", model_name)
cat("\n####################################################################################")

# get data into "timesteps form"
X_train <- build_X(trend_train, lstm_num_timesteps) 
y_train <- build_y(trend_train, lstm_num_timesteps) 

X_test <- build_X(trend_test, lstm_num_timesteps) 
y_test <- build_y(trend_test, lstm_num_timesteps) 

# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
X_train <- reshape_X_3d(X_train)
X_test <- reshape_X_3d(X_test)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]

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

# predict

pred_train <- model %>% predict(X_train, batch_size = batch_size)
pred_test <- model %>% predict(X_test, batch_size = batch_size)

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps), pred_train, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps), pred_test))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

test_rmse <- rmse(tail(trend_test,length(trend_test) - lstm_num_timesteps), pred_test)
cat("\n###########################################")
cat("\nRMSE on out-of-range test set: ", test_rmse)
cat("\n###########################################")

# test on in-range test set
trend_test <- trend_test_inrange
X_test <- build_X(trend_test, lstm_num_timesteps) 
y_test <- build_y(trend_test, lstm_num_timesteps) 
X_test <- reshape_X_3d(X_test)

pred_test <- model %>% predict(X_test, batch_size = batch_size)
df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps), pred_train, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps), pred_test))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

test_rmse <- rmse(tail(trend_test,length(trend_test) - lstm_num_timesteps), pred_test)

cat("\n###########################################")
cat("\nRMSE on in-range test set: ", test_rmse)
cat("\n###########################################")


