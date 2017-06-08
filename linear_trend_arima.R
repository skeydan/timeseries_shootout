source("common.R")

# fit 
fit <- auto.arima(trend_train)

# 1-step-ahead forecast
# re-estimate the model as new data arrives, as per https://robjhyndman.com/hyndsight/rolling-forecasts/ 
h <- 1
n <- length(trend_test) - h + 1
order <- arimaorder(fit)
predictions <- matrix(0, nrow=n, ncol=h)
lower <- matrix(0, nrow=n, ncol=h) # 95% prediction interval
upper <- matrix(0, nrow=n, ncol=h)
for(i in 1:n) {  
  x <- c(trend_train[(1+i):length(trend_train)], trend_test[1:i])
  refit <- Arima(x, order=order[1:3], seasonal=order[4:6])
  predictions[i,] <- forecast(refit, h=h)$mean
  lower[i,] <- unclass(forecast(refit, h=h)$lower)[,2]
  upper[i,] <- unclass(forecast(refit, h=h)$upper)[,2]
}

(test_rsme <- sqrt(sum((trend_test - predictions)^2))) 

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 fitted = c(fit$fitted, rep(NA, length(trend_test))),
                 preds = c(rep(NA, length(trend_train)), predictions),
                 lower = c(rep(NA, length(trend_train)), lower),
                 upper = c(rep(NA, length(trend_train)), upper))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1)


# test on in-range test set
trend_test <- trend_test_inrange
order <- arimaorder(fit)
# fit on the test set
refit <- Arima(trend_test, order=order[1:3], seasonal=order[4:6])
predictions <- refit$fitted
(test_rsme <- sqrt(sum((trend_test - predictions)^2))) 

df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test),
                 fitted = c(fit$fitted, rep(NA, length(trend_test))),
                 preds = c(rep(NA, length(trend_train)), predictions))
df <- df %>% gather(key = 'type', value = 'value', train:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) 


# n-step-ahead forecast
# tbd! #######################################################