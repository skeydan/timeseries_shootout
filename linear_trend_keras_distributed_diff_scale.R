source("common.R")
source("functions.R")

model_exists <- TRUE

lstm_num_predictions <- 4
lstm_num_timesteps <- 4 
batch_size <- 1
epochs <- 500
lstm_units <- 4
model_type <- "model_lstm_time_distributed"
lstm_type <- "stateless"
data_type <- "data_diffed_scaled"
test_type <- "TREND"

model_name <- build_model_name(model_type, test_type, lstm_type, data_type, epochs)

cat("\n####################################################################################")
cat("\nRunning model: ", model_name)
cat("\n####################################################################################")

trend_train_diff <- diff(trend_train)
trend_test_diff <- diff(trend_test)

# normalize
minval <- min(trend_train_diff)
maxval <- max(trend_train_diff)

trend_train_diff <- normalize(trend_train_diff, minval, maxval)
trend_test_diff <- normalize(trend_test_diff, minval, maxval)

train_matrix <- build_matrix(trend_train_diff, lstm_num_timesteps + lstm_num_predictions) 
test_matrix <- build_matrix(trend_test_diff, lstm_num_timesteps + lstm_num_predictions) 

X_train <- train_matrix[ ,1:4]
y_train <- train_matrix[ ,5:8]

X_test <- test_matrix[ ,1:4]
y_test <- test_matrix[ ,5:8]


# Keras LSTMs expect the input array to be shaped as (no. samples, no. time steps, no. features)
X_train <- reshape_X_3d(X_train)
X_test <- reshape_X_3d(X_test)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]

y_train <- reshape_X_3d(y_train)
y_test <- reshape_X_3d(y_test)

# model
if (!model_exists) {
  set.seed(22222)
  model <- keras_model_sequential() 
  model %>% 
    layer_lstm(units = lstm_units, input_shape = c(num_steps, num_features),
               return_sequences = TRUE) %>% 
    time_distributed(layer_dense(units = 1)) %>% 
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

# undiff
trend_train_add <- trend_train[(lstm_num_timesteps+1):(length(trend_train)-1)]
trend_train_add_matrix <- build_matrix(trend_train_add, lstm_num_predictions)
pred_train_undiff <- trend_train_add_matrix + pred_train[ , , 1]

trend_test_add <- trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]
trend_test_add_matrix <- build_matrix(trend_test_add, lstm_num_predictions)
pred_test_undiff <- trend_test_add_matrix + pred_test[ , , 1]


df <- data_frame(time_id = 1:20,
                 test = trend_test)
for(i in seq_len(nrow(pred_test))) {
  varname <- paste0("pred_test", i)
  df <- mutate(df, !!varname := c(rep(NA, lstm_num_timesteps+1),
                                  rep(NA, i-1),
                                  pred_test_undiff[i, ],
                                  rep(NA, 12-i)))
}

df <- df %>% gather(key = 'type', value = 'value', test:pred_test12)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type, linetype=type)) 


#######################################################################################
# test on in-range dataset

trend_test <- trend_test_inrange
trend_test_diff <- diff(trend_test)
trend_test_diff <- normalize(trend_test_diff, minval, maxval)

test_matrix <- build_matrix(trend_test_diff, lstm_num_timesteps + lstm_num_predictions) 

X_test <- test_matrix[ ,1:4]
y_test <- test_matrix[ ,5:8]

X_test <- reshape_X_3d(X_test)
y_test <- reshape_X_3d(y_test)

pred_test <- model %>% predict(X_test, batch_size = 1)
pred_test <- denormalize(pred_test, minval, maxval)

trend_test_add <- trend_test[(lstm_num_timesteps+1):(length(trend_test)-1)]
trend_test_add_matrix <- build_matrix(trend_test_add, lstm_num_predictions)
pred_test_undiff <- trend_test_add_matrix + pred_test[ , , 1]

df <- data_frame(time_id = 1:20,
                 test = trend_test)
for(i in seq_len(nrow(pred_test))) {
  varname <- paste0("pred_test", i)
  df <- mutate(df, !!varname := c(rep(NA, lstm_num_timesteps+1), rep(NA, i-1), pred_test_undiff[i, ], rep(NA, 12-i)))
}

df <- df %>% gather(key = 'type', value = 'value', test:pred_test12)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type, linetype=type)) 


