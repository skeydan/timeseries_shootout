library(ggplot2)
library(dplyr)
library(tidyr)
library(forecast)

source("functions.R")

traffic_df <- read_csv("internet-traffic-data-in-bits-fr.csv", col_names = c("hour", "bits"), skip = 1)
ggplot(traffic_df, aes(x = hour, y = bits)) + geom_line() + ggtitle("Internet traffic")

train <- traffic_df$bits[1:800]
test <- traffic_df$bits[801:nrow(traffic_df)]


####################################################################################
# ARIMA 1-step with confints
####################################################################################

fit <- auto.arima(train)
fit
preds_list <- forecast_rolling(fit, 1, train, test)

(test_rmse <- rmse(test, preds_list$predictions))

df <- data_frame(
                time_id = 1:num_all,
                 train_ = c(train, rep(NA, length(test))),
                 test_ = c(rep(NA, length(train)), test),
                 fitted = c(fit$fitted, rep(NA, length(test))),
                 preds = c(rep(NA, length(train)), preds_list$predictions),
                 lower = c(rep(NA, length(train)), preds_list$lower),
                 upper = c(rep(NA, length(train)), preds_list$upper))
df <- df %>% gather(key = 'type', value = 'value', train_:preds)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(color = type)) + geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1) +
  coord_cartesian(xlim = c(num_train+1, num_all))


####################################################################################
# ARIMA multistep with reestimation every 4 steps
####################################################################################

preds_list <- forecast_rolling(fit, 4, train, test)
pred_test <- drop(preds_list$predictions)
dim(pred_test)

df <- data_frame(time_id = 1:24,
                 test = test)
for(i in seq_len(nrow(pred_test))) {
  varname <- paste0("pred_test", i)
  df <- mutate(df, !!varname := c(rep(NA, i-1),
                                  pred_test[i, ],
                                  rep(NA, 21-i)))
}

df <- df %>% gather(key = 'type', value = 'value', -time_id)
ggplot(df, aes(x = time_id, y = value)) + geom_line(aes(colour = type)) 

test_matrix <- build_matrix(test,4)
(test_rmse <- rmse(test_matrix, preds_list$predictions))
