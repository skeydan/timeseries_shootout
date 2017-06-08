source("common.R")

# fit 
fit <- auto.arima(seasonal_train)

# 1-step-ahead forecast
# re-estimate the model as new data arrives, as per https://robjhyndman.com/hyndsight/rolling-forecasts/ 
h <- 1
n <- length(seasonal_test) - h + 1
fit <- auto.arima(seasonal_train)
order <- arimaorder(fit)
refit <- Arima(seasonal_test, order=order[1:3], seasonal=order[4:6])
predictions <- refit$fitted
(test_rsme <- sqrt(sum((seasonal_test - predictions)^2))) 

df <- data_frame(time_id = 1:112,
                 train = c(seasonal_train, rep(NA, length(seasonal_test))),
                 test = c(rep(NA, length(seasonal_train)), seasonal_test),
                 fitted = c(fit$fitted, rep(NA, length(seasonal_test))),
                 preds = c(rep(NA, length(seasonal_train)), predictions))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) 

# n-step-ahead forecast
# tbd! #######################################################