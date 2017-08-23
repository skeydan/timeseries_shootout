source("common.R")
source("functions.R")

cat("\n####################################################################################")
cat("\nRunning model: ", "ARIMA_seasonal_1step")
cat("\n####################################################################################")

# fit 
fit <- auto.arima(seasonal_train)
fit

# 1-step-ahead forecast
preds_list <- forecast_rolling(fit, 1, seasonal_train, seasonal_test)

df <- data_frame(time_id = 1:112,
                 train = c(seasonal_train, rep(NA, length(seasonal_test))),
                 test = c(rep(NA, length(seasonal_train)), seasonal_test),
                 fitted = c(fit$fitted, rep(NA, length(seasonal_test))),
                 preds = c(rep(NA, length(seasonal_train)), preds_list$predictions),
                 lower = c(rep(NA, length(seasonal_train)), preds_list$lower),
                 upper = c(rep(NA, length(seasonal_train)), preds_list$upper))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1)


test_rsme <- sqrt(sum((seasonal_test - preds_list$predictions)^2))
cat("\n###########################################")
cat("\nRSME on test set: ", test_rsme)
cat("\n###########################################")
