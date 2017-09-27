source("common.R")
source("functions.R")

cat("\n####################################################################################")
cat("\nRunning model: ", "ARIMA_trend_multistep")
cat("\n####################################################################################")

# fit 
fit <- auto.arima(trend_train)
fit

# 4-step-ahead forecast
preds_list <- forecast_rolling(fit, 4, trend_train, trend_test)
pred_test <- drop(preds_list$predictions)
dim(pred_test)

df <- data_frame(time_id = 1:20,
                 test = trend_test)
for(i in seq_len(nrow(pred_test))) {
  varname <- paste0("pred_test", i)
  df <- mutate(df, !!varname := c(rep(NA, i-1),
                                  pred_test[i, ],
                                  rep(NA, 17-i)))
}
calc_multiple_rmse <- function(df) {
  m <- as.matrix(df)
  ground_truth <-m[ ,2]
  pred_cols <- m[ , 8:19]
  rowwise_squared_error_sums <- apply(pred_cols, 2, function(col) sum((col - ground_truth)^2, na.rm = TRUE))
  sqrt(sum(rowwise_squared_error_sums)/length(rowwise_squared_error_sums))
}

multiple_rmse <- calc_multiple_rmse(df)

df <- df %>% gather(key = 'type', value = 'value', -time_id)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(colour = type)) 



# skip in-range test set