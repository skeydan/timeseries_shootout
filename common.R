reticulate::py_config()
reticulate::use_condaenv("r-tensorflow", required = TRUE)
require(keras)
require(forecast)
require(dplyr)
require(ggplot2)
require(readr)
require(gridExtra)
require(ggfortify)
require(tidyr)
require(lubridate)

set.seed(7777)
trend_train <- 11:110 + rnorm(100, sd = 2)
trend_test <- 111:130 + rnorm(20, sd =2)
trend_train_df <- data_frame(time_id = 1:120,
                 train = c(trend_train, rep(NA, length(trend_test))),
                 test = c(rep(NA, length(trend_train)), trend_test))
trend_train_df <- trend_train_df %>% gather(key = 'train_test', value = 'value', -time_id)
ggplot(trend_train_df, aes(x = time_id, y = value, color = train_test)) + geom_line()

set.seed(7777)
trend_test_inrange <- 31:50 + rnorm(20, sd =2)

set.seed(7777)
seasonal_train <- rep(1:7, times = 13) + rnorm(91, sd=0.2)
seasonal_test <- rep(1:7, times = 3) + rnorm(21, sd=0.2)
df <- data_frame(time_id = 1:112,
                 train = c(seasonal_train, rep(NA, length(seasonal_test))),
                 test = c(rep(NA, length(seasonal_train)), seasonal_test))
df <- df %>% gather(key = 'train_test', value = 'value', -time_id)
ggplot(df, aes(x = time_id, y = value, color = train_test)) + geom_line()

