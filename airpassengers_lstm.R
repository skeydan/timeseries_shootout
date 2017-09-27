library(ggplot2)
library(dplyr)
library(tidyr)

source("functions.R")
data("AirPassengers")

autoplot(AirPassengers)

alldata <- AirPassengers
airp_train <- window(AirPassengers,end=1958.99)
airp_test <- window(AirPassengers, start = 1959)

num_train <- length(airp_train)
num_test <- length(airp_test)
num_all <- num_train + num_test


model_exists <- FALSE

lstm_num_timesteps <- 12
batch_size <- 1
epochs <- 500
lstm_units <- 32
model_type <- "model_lstm_simple"
lstm_type <- "stateless"
data_type <- "data_diffed_scaled"
test_type <- "AIRP"

model_name <- build_model_name(model_type, test_type, lstm_type, data_type, epochs)

cat("\n####################################################################################")
cat("\nRunning model: ", model_name)
cat("\n####################################################################################")

train_diff <- diff(airp_train)[!is.na(diff(airp_train))]
test_diff <- diff(airp_test)[!is.na(diff(airp_test))]

# normalize
minval <- min(train_diff)
maxval <- max(train_diff)

train_diff <- normalize(train_diff, minval, maxval)
test_diff <- normalize(test_diff, minval, maxval)

X_train <- build_X(train_diff, lstm_num_timesteps) 
y_train <- build_y(train_diff, lstm_num_timesteps) 

X_test <- build_X(test_diff, lstm_num_timesteps) 
y_test <- build_y(test_diff, lstm_num_timesteps) 

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

pred_train <- denormalize(pred_train, minval, maxval)
pred_test <- denormalize(pred_test, minval, maxval)

pred_train_undiff <- pred_train + airp_train[(lstm_num_timesteps+1):(length(airp_train)-1)]
pred_test_undiff <- pred_test + airp_test[(lstm_num_timesteps+1):(length(airp_test)-1)]

df <- data_frame(
                 time_id = 1:144,
                 train = c(airp_train, rep(NA, length(airp_test))),
                 test = c(rep(NA, length(airp_train)), airp_test),
                 pred_train = c(rep(NA, lstm_num_timesteps+1), pred_train_undiff, rep(NA, length(airp_test))),
                 pred_test = c(rep(NA, length(airp_train)), rep(NA, lstm_num_timesteps+1), pred_test_undiff)
   )
df <- df %>% gather(key = 'type', value = 'value', train:pred_test)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type))

(test_rmse <- rmse(tail(test,length(airp_test) - lstm_num_timesteps - 1), pred_test_undiff))
