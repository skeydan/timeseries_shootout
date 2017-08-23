
############################################
#      1 step ahead forecasts              #
############################################

source("linear_trend_arima.R")
source("linear_trend_keras_simple.R")
source("linear_trend_keras_simple_diff.R")
source("linear_trend_keras_simple_diff_scale.R")

source("seasonal_arima.R")
source("seasonal_keras_simple.R")
source("seasonal_keras_simple_diff.R")
source("seasonal_keras_simple_diff_scale.R")

# use keras stateful LSTM - linear trend only
source("linear_trend_keras_stateful.R")
source("linear_trend_keras_stateful_diff.R")
source("linear_trend_keras_stateful_diff_scale.R")

############################################
#      multi step ahead forecasts          #
############################################

# arima
source("linear_trend_arima_multistep.R")
source("seasonal_arima_multistep.R")

# just diffed & scaled data, just stateless 
source("linear_trend_keras_distributed_diff_scale.R")
source("seasonal_keras_distributed_diff_scale.R")



# TBD
# arima 4 all
# for real, always use diff & scale
#
#

