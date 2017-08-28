source("common.R")
source("functions.R")

cat("\n####################################################################################")
cat("\nRunning model: ", "ARIMA_seasonal_multistep")
cat("\n####################################################################################")

# fit 
fit <- auto.arima(seasonal_train)
fit

# 6-step-ahead forecast
preds_list <- forecast_rolling(fit, 6, seasonal_train, seasonal_test)
pred_test <- drop(preds_list$predictions)
dim(pred_test)

df <- data_frame(time_id = 1:21,
                 test = seasonal_test)
for(i in seq_len(nrow(pred_test))) {
  varname <- paste0("pred_test", i)
  df <- mutate(df, !!varname := c(rep(NA, i-1),
                                  pred_test[i, ],
                                  rep(NA, 16-i)))
}

df <- df %>% gather(key = 'type', value = 'value', -time_id)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(colour = type)) 

test_matrix <- build_matrix(seasonal_test,6)
(test_rmse <- rmse(test_matrix, preds_list$predictions))

# skip in-range test set