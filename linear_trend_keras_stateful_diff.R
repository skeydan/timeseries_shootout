source("common.R")
source("functions.R")

model_exists <- FALSE

# since we are using stateful rnn tsteps can be set to 1
lstm_num_timesteps <- 1
batch_size <- 1
epochs <- 500
lstm_units <- 4
model_type <- "model_lstm_simple"
lstm_type <- "stateful"
data_type <- "data_diffed"
test_type <- "TREND"

model_name <- build_model_name(model_type, test_type, lstm_type, data_type, epochs)

cat("\n####################################################################################")
cat("\nRunning model: ", model_name)
cat("\n####################################################################################")

# difference
trend_train_start <- trend_train[1]
trend_test_start <- trend_test[1]

trend_train_diff <- diff(trend_train)
trend_test_diff <- diff(trend_test)

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

model %>% reset_states()
pred_train <- model %>% predict(X_train, batch_size = batch_size)
model %>% reset_states()
pred_test <- model %>% predict(X_test, batch_size = batch_size)

pred_train_undiff <- pred_train + trend_train[(lstm_num_timesteps+1):(length(trend_train)-1)]
pred_test_undiff <- pred_test + trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

test_rsme <- sqrt(sum((tail(trend_test,length(trend_test) - lstm_num_timesteps) - pred_test)^2))
cat("\n###########################################")
cat("\nRSME on out-of-range test set: ", test_rsme)
cat("\n###########################################")

# test on in-range test set
trend_test <- trend_test_inrange
trend_test_diff <- diff(trend_test)
X_test <- build_X(trend_test_diff, lstm_num_timesteps) 
y_test <- build_y(trend_test_diff, lstm_num_timesteps) 
X_test <- reshape_X_3d(X_test)

model %>% reset_states()
pred_test <- model %>% predict(X_test, batch_size = batch_size)
pred_test_undiff <- pred_test + trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(trend_test))),
                 pred_test = c(rep(NA, length(trend_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

test_rsme <- sqrt(sum((tail(trend_test,length(trend_test) - lstm_num_timesteps) - pred_test)^2))

cat("\n###########################################")
cat("\nRSME on in-range test set: ", test_rsme)
cat("\n###########################################")
