source("common.R")
source("functions.R")

model_exists <- TRUE

lstm_num_timesteps <- 6
batch_size <- 1
epochs <- 500
lstm_units <- 4
lstm_type <- "stateless"
data_type <- "data_diffed"
test_type <- "SEASONAL"
model_type <- "model_lstm_simple"

model_name <- build_model_name(model_type, test_type, lstm_type, data_type, epochs)

cat("\n####################################################################################")
cat("\nRunning model: ", model_name)
cat("\n####################################################################################")

seasonal_train_diff <- diff(seasonal_train)
seasonal_test_diff <- diff(seasonal_test)

# get data into "timesteps form"
X_train <- build_X(seasonal_train_diff, lstm_num_timesteps) 
y_train <- build_y(seasonal_train_diff, lstm_num_timesteps) 

X_test <- build_X(seasonal_test_diff, lstm_num_timesteps) 
y_test <- build_y(seasonal_test_diff, lstm_num_timesteps) 

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

pred_train <- model %>% predict(X_train, batch_size = 1)
pred_test <- model %>% predict(X_test, batch_size = 1)
pred_train_undiff <- pred_train + seasonal_train[(lstm_num_timesteps+1):(length(seasonal_train)-1)]
pred_test_undiff <- pred_test + seasonal_test[(lstm_num_timesteps+1):(length(seasonal_test)-1)]

df <- data_frame(time_id = 1:112,
                 train = c(seasonal_train, rep(NA, length(seasonal_test))),
                 test = c(rep(NA, length(seasonal_train)), seasonal_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(seasonal_test))),
                 pred_test = c(rep(NA, length(seasonal_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff))
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

test_rsme <- sqrt(sum((tail(seasonal_test,length(seasonal_test) - lstm_num_timesteps - 1) - pred_test_undiff)^2)) 
cat("\n###########################################")
cat("\nRSME on test set: ", test_rsme)
cat("\n###########################################")
