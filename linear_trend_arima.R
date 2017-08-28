source("common.R")
source("functions.R")

cat("\n####################################################################################")
cat("\nRunning model: ", "ARIMA_trend_1step")
cat("\n####################################################################################")

# fit 
fit <- auto.arima(trend_train)
fit

# 1-step-ahead forecast
preds_list <- forecast_rolling(fit, 1, trend_train, trend_test)

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 fitted = c(fit$fitted, rep(NA, length(trend_test))),
                 preds = c(rep(NA, length(trend_train)), preds_list$predictions),
                 lower = c(rep(NA, length(trend_train)), preds_list$lower),
                 upper = c(rep(NA, length(trend_train)), preds_list$upper))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1)


test_rmse <- rmse(trend_test, preds_list$predictions)
cat("\n###########################################")
cat("\nRMSE on out-of-range test set: ", test_rmse)
cat("\n###########################################")


# test on in-range test set
# makes less sense here, as we assume the same process
trend_test <- trend_test_inrange
order <- arimaorder(fit)
# re-fit on the test set
refit <- Arima(trend_test, order=order[1:3], seasonal=order[4:6])
# fitted values are simply one-step forecasts
predictions <- refit$fitted

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 fitted = c(fit$fitted, rep(NA, length(trend_test))),
                 preds = c(rep(NA, length(trend_train)), predictions))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) 


test_rmse <- rmse(trend_test, predictions)
cat("\n###########################################")
cat("\nRMSE on in-range test set: ", test_rmse)
cat("\n###########################################")
