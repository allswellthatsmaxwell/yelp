
library(forecast)
library(magrittr)


#' get n lags from point t (n <= t) in y
#' (i.e. each of y[t-1], y[t-2], ..., y[n+1], y[n])
get_lags <- function(y, t, n) {
  if (n > t) stop (glue("must have n <= t (got n = {n}, t = {t})"))
  lags <- sapply(1:n, function(i) y[t - i])
  matrix(lags, nrow = 1)
}

#' expands y into a (length(y) - n)-by-(n + 1) matrix,
#' where for j == 1, m[i,j] is y[i] and for j > 1, m[i, j] is
#' the jth lag of y[i].
#' @param y a timeseries
#' @param n a scalar int: number of lags to capture
get_n_lag_matrix <- function(y, n) {
  ylen <- length(y)
  lapply((n + 1):ylen,
         function(t) cbind(y[t], get_lags(y, t, n))) %>%
    do.call(rbind, .)
}


#' Train and return an xgboost model where the response
#' variable is y[i] and the input features are n lags of
#' y. y must already be ordered properly
train_xg <- function(y, nlags) {
  trn_lag_mat <- get_n_lag_matrix(y, nlags)
  trn_response <- trn_lag_mat[,1]
  trn_predictors <- trn_lag_mat[,-1]
  xgboost(trn_predictors, trn_response, nrounds = 100)
}

#' Predict using lags as predictors. Where known lags don't exist because
#' it is the future, predicted values are used instead.
#' This is accomplished by using the ith prediction as the
#' 1st lag on the (i + 1)th prediction step, the (i-1)th prediction
#' as the second lag, and so on.
#' @param model a model with a predict function that was
#' trained on nlags columns (one column per lag) (currently
#' maybe only xgboost works)
#' @param y the same y used to train the model
#' @param horizon the number of period to predict
#' @param nlags the number of lags used to train the model
walk_prediction <- function(model, y, horizon, nlags) {
  ## The first prediction uses nlag true values from y;
  ## each subsequent prediction step throws one true value out
  ## from early in y and tacks the predicted value onto the
  ## end, and this happens horizon times. so we'll eventually be
  ## using nlags + horizon values from this vector (but only nlags
  ## at a time).
  short_y <- rep(as.numeric(NA), nlags + horizon)
  short_y[1:nlags] <- take_last_n(y, nlags)
  for (i in 1:horizon) {
    t_i <- nlags + i
    lags_for_predict <- get_lags(short_y, t_i, nlags)
    short_y[t_i] <- predict(model, newdata = lags_for_predict)
  }
  take_last_n(short_y, horizon)
}


#' generate dates from test_start to horizon days after test_start
generate_test_ds <- function(test_start, horizon) {
  seq.Date(from = test_start,
           to = (test_start + horizon - 1),
           by = 1)
}

#' get daily errors between true and predicted values
#' @param true_dat, pred_dat dataframes with the columns ds and y
#' @return a dataframe with one record per day and
#' the columns ds and y_minus_yhat
get_daily_errors <- function(true_dat, pred_dat) {
  inner_join(select(true_dat, ds, y),
             select(pred_dat, ds, y),
             by = "ds",
             suffix = c("", "hat")) %>%
    mutate(y_minus_yhat = y - yhat)
}


#' get the in-sample fit for y from model
get_fit <- function(model, y, nlags) {
  trn_lag_mat <- get_n_lag_matrix(y, nlags)
  trn_response <- trn_lag_mat[,1]
  trn_predictors <- trn_lag_mat[,-1]
  yfit <- predict(model, trn_predictors)
  tibble(ds = seq.Date(from = train_start + nlags,
                       to = train_end,
                       by = 1),
         y = yfit)
}

take_last_n <- function(vec, n) vec[(length(vec) - n + 1):length(vec)]

roll_validation <- function(dat, earliest_date, horizon) {

}

STATE <- "Arizona"

one_state_dat <- state_review_values_by_date %>%
  prepare_for_state_modeling() %>%
  filter(state == STATE) %>%
  ungroup() %>%
  select(-state)

train_start <- min(one_state_dat$ds)
train_end <- as.Date("2016-06-30")
test_start <- train_end + 1
test_end <- max(one_state_dat$ds)

ds_all <-
  tibble(ds = seq.Date(from = train_start, to = test_end, by = 1),
         period = case_when(between(ds, train_start, train_end) ~ "train",
                            between(ds, test_start, test_end) ~ "test"))

one_state_dat_complete <- one_state_dat %>%
  right_join(ds_all, by = "ds") %>%
  ## The missing dates are all early and in the time of low activity,
  ## and it seems like Yelp has provided all non-zero days,
  ## so assume missing days are zeros.
  mutate(y = ifelse(is.na(y), 0, y)) %>%
  mutate(type = "actual")

one_state_dat_complete %<>% mutate(y = y - lag(y))

trn <- one_state_dat_complete %>% filter(period == "train")
tst <- one_state_dat_complete %>% filter(period == "test")
y_trn <- trn %>% arrange(ds) %$% ts(y)
y_tst <- tst %>% arrange(ds) %$% ts(y)

####
## Neural net (FF, single hidden layer... poor default performance) ############
####

model <- nnetar(y_trn)

fit_dat <- tibble(ds = ds_all %>% filter(period == "train") %$% ds,
                  y = model$fitted,
                  type = "fitted")

fcast <- forecast(model, h = ds_all %>% filter(period == "test") %>% nrow)
fcast_dat <- tibble(ds = ds_all %>% filter(period == "test") %$% ds,
                    y = fcast$mean,
                    type = "predicted")

prophet_result <- stacked_accuracies %>%
  filter(state == STATE) %>%
  ungroup() %>%
  select(ds, yhat, group) %>%
  rename(type = group, y = yhat) %>%
  filter(type == WITH_HOLS_NAME)

fcast_dat %>%
  bind_rows(fit_dat, one_state_dat_complete, prophet_result) %>%
  filter(year(ds) >= 2015) %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_point(alpha = 0.2) +
  theme_bw()

######
## Single-state prophet. #######################################################
######

PHLABEL <- "Prophet"
proph <- prophet(trn)
prophet_preds_frame <-
  predict(proph, tibble(ds = generate_test_ds(test_start, horizon))) %>%
  mutate(ds = as.Date(ds), type = PHLABEL) %>%
  rename(y = yhat) %>%
  select(ds, y, type)

######
## Forecasting with ensemble model. ############################################
######

horizon <- 365
NLAGS <- 365 * 2

test_set_x_scale <- list(scale_x_date(date_breaks = "2 weeks",
                                      labels = function(d) format(d, "%d %b %Y")))

XGLABEL <- "xgboost"

y_trn <- trn %>% arrange(ds) %$% y
xg <- train_xg(y_trn, NLAGS)
xg_preds <- walk_prediction(xg, y_trn, horizon, NLAGS)

xg_preds_frame <- tibble(ds = generate_test_ds(test_start, horizon),
                         y = xg_preds,
                         type = XGLABEL)

xg_fit_frame <- get_fit(xg, y_trn, NLAGS)

.xg_fit_plot <- xg_fit_frame %>%
  mutate(type = XGLABEL) %>%
  bind_rows(trn) %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_point(alpha = 0.3, size = 4) +
  theme_bw() +
  scale_x_date(date_breaks = "1 year", labels = function(d) format(d, "%Y")) +
  theme(legend.title = element_blank(),
        legend.text = element_text(size = 30),
        aspect.ratio = 2/5) +
  labs(x = "Date", y = DAILY_REVIEWS_YLAB)



.xg_prophet_comparison_plot <- bind_rows(trn,
                                         tst,
                                         xg_preds_frame,
                                         prophet_preds_frame) %>%
  filter(ds <= test_start + horizon - 1) %>%
  filter(ds > test_start - 0 * horizon) %>%
  ggplot(aes(x = ds, y = y, color = type)) +
  geom_line() +
  theme_bw() +
  test_set_x_scale +
  labs(x = "Date", y = DAILY_REVIEWS_YLAB) +
  theme(legend.title = element_blank(),
        aspect.ratio = 2/5)

xg_errors <- get_daily_errors(tst, xg_preds_frame) %>%
  mutate(type = XGLABEL)
ph_errors <- get_daily_errors(tst, prophet_preds_frame) %>%
  mutate(type = PHLABEL)

bind_rows(xg_errors, ph_errors) %>%
  ggplot(aes(x = ds, y = y_minus_yhat, color = type)) +
  geom_point() +
  theme_bw() +
  geom_hline(yintercept = 0)

better_by_day <-
  inner_join(xg_errors, ph_errors, by = "ds", suffix = c("_xg", "_ph")) %>%
  mutate(better = case_when(abs(y_minus_yhat_xg) < abs(y_minus_yhat_ph) ~
                              as.character("xgboost"),
                            abs(y_minus_yhat_xg) > abs(y_minus_yhat_ph) ~
                              PHLABEL,
                            TRUE ~ "Same"),
         error_difference = abs(y_minus_yhat_ph) - abs(y_minus_yhat_xg),
         cumu_error_diff = cumsum(error_difference) / 1:n())


.better_by_day_plot <- better_by_day %>%
  ggplot(aes(x = ds)) +
  geom_point(aes(y = error_difference, color = better), size = 5, alpha = 0.9) +
  geom_line(aes(y = cumu_error_diff), size = 1.2) +
  geom_hline(yintercept = 0, color = "purple") +
  theme_bw() +
  labs(x = "Date",
       y = "Prophet error - xgboost error (in reviews per day)",
       title = glue("Difference in error by day between xgboost with {NLAGS} lags, and Prophet"),
       subtitle = glue("Days are colored by which model has the smaller absolute error.
The black line is the cumulative better-ness of xgboost.")) +
  theme(legend.title = element_text(size = 30),
        aspect.ratio = 2/5) +
  test_set_x_scale
