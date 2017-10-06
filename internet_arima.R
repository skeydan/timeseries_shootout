library(ggplot2)
library(dplyr)
library(tidyr)
library(forecast)
library(readr)

source("functions.R")

traffic_df <- read_csv("internet-traffic-data-in-bits-fr.csv", col_names = c("hour", "bits"), skip = 1)
ggplot(traffic_df, aes(x = hour, y = bits)) + geom_line() + ggtitle("Internet traffic")

train <- traffic_df$bits[1:800]
test <- traffic_df$bits[801:nrow(traffic_df)]

# 1st try
fit <- auto.arima(train, trace = TRUE)
fit

# 2nd
train_ts <- msts(train,seasonal.periods = c(24, 24*7))
fit <- auto.arima(train_ts, trace = TRUE)

# 3rd
#fit <- auto.arima(train_ts, stepwise = FALSE, trace = TRUE)
                  
# 4th
fit <- tbats(train_ts)
fit

# forecast
# plot(forecast(fit, h=14*24))
                  
test_ts <- msts(test,seasonal.periods = c(24, 24*7))

n_forecast <- 1
n <- length(test_ts) - n_forecast + 1
predictions <- matrix(0, nrow=n, ncol= n_forecast)
lower <- matrix(0, nrow=n, ncol= n_forecast) 
upper <- matrix(0, nrow=n, ncol= n_forecast)
  
for(i in 1:n) {  
  print(i)
  x <- c(train_ts, test_ts[0:(i-1)])
  x <- msts(x,seasonal.periods = c(24, 24*7))
  refit <- tbats(x)
  predictions[i,] <- forecast(refit, h = n_forecast)$mean
  lower[i,] <- unclass(forecast(refit, h = n_forecast)$lower)[,2] # 95% prediction interval
  upper[i,] <- unclass(forecast(refit, h = n_forecast)$upper)[,2] # 95% prediction interval
  print(predictions[i,])
}
  
preds_list <- list(predictions = predictions, lower = lower, upper = upper)

(test_rmse <- rmse(test, preds_list$predictions))

df <- data_frame(
                time_id = 1:1231,
                 train_ = c(train, rep(NA, length(test))),
                 test_ = c(rep(NA, length(train)), test),
                 fitted = c(fit$fitted, rep(NA, length(test))),
                 preds = c(rep(NA, length(train)), preds_list$predictions),
                 lower = c(rep(NA, length(train)), preds_list$lower),
                 upper = c(rep(NA, length(train)), preds_list$upper))
df <- df %>% gather(key = 'type', value = 'value', train_:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1) +
  coord_cartesian(xlim = c(length(train) +1, length(train) + length(test) ))

