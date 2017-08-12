
############################################
#      1 step ahead forecasts              #
############################################

source("linear_trend_keras_simple.R")
source("linear_trend_keras_simple_diff.R")
source("linear_trend_keras_simple_diff_scale.R")

source("seasonal_keras_simple.R")
source("seasonal_keras_simple_diff.R")
source("seasonal_keras_simple_diff_scale.R")

# use keras stateful LSTM
source("linear_trend_keras_stateful.R")
source("linear_trend_keras_stateful_diff.R")
source("linear_trend_keras_stateful_diff_scale.R")

# not implemented (yet?) 
# stateful implementations for seasonal data



############################################
#      multi step ahead forecasts          #
############################################

source("linear_trend_keras_distributed_diff_scale.R")
source("seasonal_keras_distributed_diff_scale.R")

# TBD
# arima 4 all
# for real, always use diff & scale
#
#

